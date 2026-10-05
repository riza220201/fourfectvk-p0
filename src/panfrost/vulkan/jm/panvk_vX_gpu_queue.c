/*
 * Copyright © 2021 Collabora Ltd.
 *
 * Derived from tu_device.c which is:
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "genxml/gen_macros.h"

#include "decode.h"

#include "panvk_cmd_buffer.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_event.h"
#include "panvk_image.h"
#include "panvk_image_view.h"
#include "panvk_instance.h"
#include "panvk_physical_device.h"
#include "panvk_priv_bo.h"
#include "panvk_queue.h"

#include "vk_drm_syncobj.h"
#include "vk_sync.h"
#include "vk_framebuffer.h"

#include "drm-uapi/panfrost_drm.h"
#include "drm-uapi/mali_kbase_ioctl.h"
#include <fcntl.h>
#include <inttypes.h>
#include <poll.h>
#include <sys/mman.h>
#include <unistd.h>
#include "pan_v9_trace.h"

struct panvk_kbase_event_v2 {
   uint32_t event_code;
   uint8_t atom_number;
   uint8_t padding[3];
   uint64_t udata[2];
};


/*
 * Dumps the first bytes of the tiler heap so a caller can diff the contents
 * before/after a submit and tell whether the tiler wrote anything at all.
 *
 * The heap is allocated with PAN_KMOD_BO_FLAG_ALLOC_ON_FAULT, which makes
 * kbase pass commit_pages=0 (see kbase_kmod.c), so the mapping exists but has
 * no pages behind it until the GPU faults them in. Dereferencing addr.host
 * directly therefore SIGSEGVs whenever the tiler did *not* write - which is
 * exactly the case we are trying to observe. Reading through /proc/self/mem
 * instead turns that into an EIO/EFAULT return value we can report.
 *
 * mincore() is only used as a hint: on some kernels it does not report
 * residency for device mappings, so a "not resident" answer there is not
 * treated as conclusive - the pread is what decides.
 */
/*
 * Where the heap dump and submit results go.
 *
 * stderr is not usable under a Winlator/Ludashi-style container: nothing
 * collects it, so the output is simply lost. Set PANVK_HEAP_LOG to a writable
 * path to capture it instead, e.g.
 *
 *    PANVK_HEAP_LOG=/sdcard/Download/panvk-heap.log
 *
 * Unset (or unopenable) means fall back to stderr, which keeps the behaviour
 * of the original patch. The file is opened once, appended to, and line
 * buffered so a crash mid-run still leaves the lines already written.
 */
static FILE *
panvk_debug_log_file(void)
{
   static FILE *log_file = NULL;
   static bool tried = false;

   if (!tried) {
      tried = true;
      const char *path = getenv("PANVK_HEAP_LOG");

      if (path && path[0]) {
         log_file = fopen(path, "ae");
         if (log_file) {
            setvbuf(log_file, NULL, _IOLBF, 0);
            fprintf(log_file, "[PANVK_DEBUG_HEAP] --- log dibuka, pid=%d ---\n",
                    (int)getpid());
         } else {
            fprintf(stderr,
                    "[PANVK_DEBUG_HEAP] gagal membuka PANVK_HEAP_LOG=%s: %s,"
                    " jatuh ke stderr\n",
                    path, strerror(errno));
         }
      }
   }

   return log_file ? log_file : stderr;
}

static void
panvk_debug_dump_tiler_heap(struct panvk_device *dev, const char *label)
{
   FILE *out = panvk_debug_log_file();

   if (!dev->tiler_heap || !dev->tiler_heap->addr.host) {
      fprintf(out, "[PANVK_DEBUG_HEAP] %s: tiler_heap belum ada/belum ke-mmap\n",
              label);
      return;
   }

   void *host = dev->tiler_heap->addr.host;
   const size_t dump_len = 64;

   fprintf(out,
           "[PANVK_DEBUG_HEAP] %s bo->flags=0x%x dev=0x%" PRIx64
           " size=%zu host=%p",
           label, dev->tiler_heap->bo->flags, dev->tiler_heap->addr.dev,
           (size_t)pan_kmod_bo_size(dev->tiler_heap->bo), host);

   unsigned char resident = 0;
   long page_size = sysconf(_SC_PAGESIZE);
   if (page_size > 0 &&
       mincore((void *)((uintptr_t)host & ~(uintptr_t)(page_size - 1)),
               page_size, &resident) == 0) {
      fprintf(out, " mincore=%s", (resident & 1) ? "resident" : "absent");
   } else {
      fprintf(out, " mincore=unavailable");
   }

   /* Safe read: never dereference host directly, see comment above. */
   uint8_t buf[64];
   int mem_fd = open("/proc/self/mem", O_RDONLY | O_CLOEXEC);
   if (mem_fd < 0) {
      fprintf(out, " -- open(/proc/self/mem) gagal: %s\n", strerror(errno));
      return;
   }

   ssize_t n = pread(mem_fd, buf, dump_len, (off_t)(uintptr_t)host);
   int saved_errno = errno;
   close(mem_fd);

   if (n < 0) {
      /* EIO here means the page is mapped but not backed: the GPU never
       * faulted it in, i.e. nothing wrote to the heap. */
      fprintf(out, " -- UNREADABLE (%s) => halaman belum ter-commit,"
                      " tiler tidak menulis apa pun\n",
              strerror(saved_errno));
      return;
   }

   bool all_zero = true;
   for (ssize_t i = 0; i < n; i++) {
      if (buf[i] != 0) {
         all_zero = false;
         break;
      }
   }

   fprintf(out, " first%zd%s:", n, all_zero ? " (ALL ZERO)" : "");
   for (ssize_t i = 0; i < n; i++)
      fprintf(out, " %02x", buf[i]);
   fprintf(out, "\n");
}

/*
 * Reads back every job header in the batch AFTER the GPU ran it.
 *
 * Job Header (v9.xml) starts with the fields the GPU itself writes:
 *
 *   Exception Status       word 0
 *   First Incomplete Task  word 1
 *   Fault Pointer          word 2..3
 *
 * The pre-submit dump always shows these as zero because nothing has run yet,
 * so a job that is silently rejected by the GPU looks identical to one that
 * succeeded. Reading them back after the wait is the only way to tell apart
 * "the GPU ran this and wrote nothing" from "the GPU refused this".
 *
 * Exception Status is a BASE_JD_EVENT_* code. Verified against the r54p1 UAPI
 * header (jm/mali_base_jm_kernel.h):
 *
 *   BASE_JD_EVENT_NOT_STARTED = 0x00
 *   BASE_JD_EVENT_DONE        = 0x01   <- success
 *   BASE_JD_EVENT_STOPPED     = 0x03
 *   BASE_JD_EVENT_TERMINATED  = 0x04
 *   BASE_JD_EVENT_ACTIVE      = 0x08
 *   ...RANGE_HW_FAULT_OR_SW_ERROR_START = 0x40
 *
 * So 0x01 is *success*, and anything below 0x40 is non-fatal. Only >= 0x40 is
 * an actual fault. Do not treat "non-zero" as an error here - an earlier
 * version of this print did and sent the investigation down a dead end.
 *
 * batch->jobs holds the CPU pointers of every job descriptor in the batch,
 * which is exactly what we need - these are pool allocations, still mapped.
 */
#define PANVK_JD_EVENT_NOT_STARTED 0x00
#define PANVK_JD_EVENT_DONE        0x01
#define PANVK_JD_EVENT_FAULT_START 0x40

static const char *
panvk_jd_event_str(uint32_t code)
{
   switch (code) {
   case PANVK_JD_EVENT_NOT_STARTED:
      return "NOT_STARTED";
   case PANVK_JD_EVENT_DONE:
      return "DONE";
   case 0x03:
      return "STOPPED";
   case 0x04:
      return "TERMINATED";
   case 0x08:
      return "ACTIVE";
   default:
      return code >= PANVK_JD_EVENT_FAULT_START ? "FAULT" : "non-fatal";
   }
}

static void
panvk_debug_dump_job_status(struct panvk_cmd_buffer *cmdbuf,
                            struct panvk_batch *batch, const char *label)
{
   FILE *out = panvk_debug_log_file();
   unsigned idx = 0;

   /* The descriptor pool is CPU-cached, so invalidate before reading back
    * anything the GPU may have written. */
   panvk_pool_invalidate_maps(&cmdbuf->desc_pool);

   util_dynarray_foreach(&batch->jobs, void *, job) {
      const uint8_t *bytes = *job;
      uint32_t exception_status, first_incomplete_task, word4, word5;
      uint64_t fault_pointer, next_job;

      memcpy(&exception_status, bytes + 0, sizeof(exception_status));
      memcpy(&first_incomplete_task, bytes + 4, sizeof(first_incomplete_task));
      memcpy(&fault_pointer, bytes + 8, sizeof(fault_pointer));
      memcpy(&word4, bytes + 16, sizeof(word4));
      /* Job Header, v9.xml: Index is word 4 bits 16..31, Dependency 1 and 2 are
       * word 5 low and high halves, Next is the 64-bit word 6. Reading these
       * back tells us the dependency graph the hardware was actually given,
       * which the job type alone does not. */
      memcpy(&word5, bytes + 20, sizeof(word5));
      memcpy(&next_job, bytes + 24, sizeof(next_job));

      fprintf(out,
              "[PANVK_DEBUG_JOBSTATUS] %s job[%u] type=%u"
              " index=%u dep1=%u dep2=%u next=0x%" PRIx64
              " exception_status=0x%02x (%s) first_incomplete_task=%u"
              " fault_pointer=0x%" PRIx64 "%s\n",
              label, idx, (word4 >> 1) & BITFIELD_MASK(7),
              (word4 >> 16) & 0xffff, word5 & 0xffff, (word5 >> 16) & 0xffff,
              next_job, exception_status,
              panvk_jd_event_str(exception_status), first_incomplete_task,
              fault_pointer,
              exception_status >= PANVK_JD_EVENT_FAULT_START
                 ? "  <== GPU FAULT"
                 : "");
      idx++;
   }
}

static int
panvk_kbase_submit_and_wait(int fd, uint64_t jc, uint32_t core_req)
{
   uint8_t atom_bytes[64];
   memset(atom_bytes, 0, sizeof(atom_bytes));
   memcpy(atom_bytes + 0, &jc, 8);
   atom_bytes[40] = 1;
   memcpy(atom_bytes + 44, &core_req, 4);

   struct kbase_ioctl_job_submit submit = {
      .addr = (uint64_t)(uintptr_t)atom_bytes,
      .nr_atoms = 1,
      .stride = 64,
   };

   if (ioctl(fd, KBASE_IOCTL_JOB_SUBMIT, &submit))
      return -errno;

   struct pollfd pfd = { .fd = fd, .events = POLLIN };
   int pret = poll(&pfd, 1, 5000);
   if (pret <= 0 || !(pfd.revents & POLLIN))
      return -110;

   struct panvk_kbase_event_v2 event;
   ssize_t n = read(fd, &event, sizeof(event));
   if (n != sizeof(event))
      return -5;

   if (event.event_code != 0x1)
      return -(int)event.event_code;

   return 0;
}

/* Kernel-side chaining. panvk_kbase_submit_and_wait() costs two CPU round trips
 * per render pass (vertex/tiler chain, then fragment chain), and the GPU idles
 * during each one: measured on the RS4 as ~0.26 ms per pass even for a pass
 * that covers a single 16x16 tile. Instead, the atoms of a whole vkQueueSubmit
 * are queued with a DATA dependency on the previous atom -- the same execution
 * order as before, enforced by kbase instead of by the CPU -- submitted with
 * one ioctl, and waited for once. Only the last atom reports success; a failed
 * atom reports itself and cancels its dependents.
 * PANVK_KBASE_SYNC_SUBMIT=1 restores the per-chain synchronous path. */
#define PANVK_KBASE_MAX_CHAIN 120
#define PANVK_KBASE_ATOM_STRIDE 64 /* base_jd_atom_v2 + MTK BM_2 frame_nr */
/* From the kbase JM UAPI, mali_base_jm_kernel.h (r54p1). */
#define BASE_JD_DEP_TYPE_DATA (1U << 0)
#define BASE_JD_REQ_EVENT_ONLY_ON_FAILURE (1U << 12)

struct panvk_kbase_chain {
   uint8_t atoms[PANVK_KBASE_MAX_CHAIN][PANVK_KBASE_ATOM_STRIDE];
   const struct panvk_batch *batches[PANVK_KBASE_MAX_CHAIN];
   unsigned nr_atoms, nr_batches;
};

static simple_mtx_t panvk_kbase_chain_mtx = SIMPLE_MTX_INITIALIZER;
static struct panvk_kbase_chain panvk_kbase_chain;

static bool
panvk_kbase_chain_enabled(void)
{
   /* The trace/dump paths decode each chain right after it ran. */
   return !getenv("PANVK_KBASE_SYNC_SUBMIT") && !pan_v9_trace_enabled() &&
          !getenv("PANVK_HEAP_LOG") && !PANVK_DEBUG(TRACE) && !PANVK_DEBUG(DUMP);
}

static void
panvk_kbase_chain_append(struct panvk_kbase_chain *c, uint64_t jc,
                         uint32_t core_req)
{
   uint8_t *a = c->atoms[c->nr_atoms];
   const uint8_t num = c->nr_atoms + 1;

   memset(a, 0, PANVK_KBASE_ATOM_STRIDE);
   memcpy(a + 0, &jc, 8);
   if (num > 1) {
      a[36] = num - 1;                 /* pre_dep[0].atom_id */
      a[37] = BASE_JD_DEP_TYPE_DATA;   /* pre_dep[0].dependency_type */
   }
   a[40] = num;                        /* atom_number */
   core_req |= BASE_JD_REQ_EVENT_ONLY_ON_FAILURE;
   memcpy(a + 44, &core_req, 4);
   c->nr_atoms++;
}

static int
panvk_kbase_chain_flush(int fd, struct panvk_kbase_chain *c)
{
   const unsigned n = c->nr_atoms;
   int err = 0;

   c->nr_atoms = 0;
   c->nr_batches = 0;
   if (!n)
      return 0;

   /* The last atom reports success too: it is the one we wait for. */
   uint32_t core_req;
   memcpy(&core_req, c->atoms[n - 1] + 44, 4);
   core_req &= ~BASE_JD_REQ_EVENT_ONLY_ON_FAILURE;
   memcpy(c->atoms[n - 1] + 44, &core_req, 4);

   struct kbase_ioctl_job_submit submit = {
      .addr = (uint64_t)(uintptr_t)c->atoms,
      .nr_atoms = n,
      .stride = PANVK_KBASE_ATOM_STRIDE,
   };
   if (ioctl(fd, KBASE_IOCTL_JOB_SUBMIT, &submit))
      return -errno;

   for (;;) {
      struct pollfd pfd = { .fd = fd, .events = POLLIN };
      int pret = poll(&pfd, 1, 20000);
      if (pret <= 0 || !(pfd.revents & POLLIN))
         return -110;

      struct panvk_kbase_event_v2 event;
      if (read(fd, &event, sizeof(event)) != sizeof(event))
         return -5;

      if (event.event_code != 0x1) {
         mesa_loge("panvk: kbase atom %u of %u failed: event 0x%x",
                   event.atom_number, n, event.event_code);
         if (!err)
            err = -(int)event.event_code;
      }
      if (event.atom_number == n)
         return err;
   }
}

static bool
panvk_kbase_chain_has_batch(const struct panvk_kbase_chain *c,
                            const struct panvk_batch *batch)
{
   for (unsigned i = 0; i < c->nr_batches; i++) {
      if (c->batches[i] == batch)
         return true;
   }
   return false;
}

static void
panvk_queue_submit_batch(struct panvk_gpu_queue *queue,
                         struct panvk_cmd_buffer *cmdbuf,
                         struct panvk_batch *batch, uint32_t *bos,
                         unsigned nr_bos, uint32_t *in_fences,
                         unsigned nr_in_fences,
                         struct panvk_kbase_chain *chain)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   struct panvk_physical_device *phys_dev =
      to_panvk_physical_device(dev->vk.physical);
   ASSERTED int ret;

   /* Reset the batch if it's already been issued */
   if (batch->issued) {
      /* Never rewrite a batch that is still queued in the current chain. */
      if (chain && panvk_kbase_chain_has_batch(chain, batch)) {
         ret = panvk_kbase_chain_flush(dev->drm_fd, chain);
         if (ret)
            mesa_loge("panvk: v9 chained submit failed: %d", ret);
      }

      util_dynarray_foreach(&batch->jobs, void *, job)
         memset((*job), 0, 4 * 4);

      /* Reset the tiler before re-issuing the batch */
      if (batch->tiler.ctx_descs.cpu) {
         memcpy(batch->tiler.heap_desc.cpu, &batch->tiler.heap_templ,
                sizeof(batch->tiler.heap_templ));

         struct mali_tiler_context_packed *ctxs = batch->tiler.ctx_descs.cpu;

         for (uint32_t i = 0; i < batch->fb.layer_count; i++)
            memcpy(&ctxs[i], &batch->tiler.ctx_templ, sizeof(*ctxs));
      }

      /* We don't keep track of BO <-> job relationship, so let's just flush the
       * whole desc pool for now. */
      panvk_pool_flush_maps(&cmdbuf->desc_pool);
   }

   /* Flush pending synchronization requests before submitting the job, to
    * make sure things are GPU-visible. */
   pan_kmod_flush_bo_map_syncs(dev->kmod.dev);

   if (chain) {
      if (chain->nr_atoms + 2 > PANVK_KBASE_MAX_CHAIN ||
          chain->nr_batches + 1 > PANVK_KBASE_MAX_CHAIN) {
         ret = panvk_kbase_chain_flush(dev->drm_fd, chain);
         if (ret)
            mesa_loge("panvk: v9 chained submit failed: %d", ret);
      }
      if (batch->vtc_jc.first_job)
         panvk_kbase_chain_append(chain, batch->vtc_jc.first_job, 0x16);
      if (batch->frag_jc.first_job)
         panvk_kbase_chain_append(chain, batch->frag_jc.first_job, 0x01);
      chain->batches[chain->nr_batches++] = batch;
      batch->issued = true;
      return;
   }

   /* SKIP: dump sebelum submit dipastikan SIGSEGV karena ALLOC_ON_FAULT (commit_pages=0), nggak informatif */
   if (batch->vtc_jc.first_job) {
      ret = panvk_kbase_submit_and_wait(dev->drm_fd, batch->vtc_jc.first_job, 0x16); /* PATCH eksperimen: dulu 0x16 (CS|T|V) */
      if (ret)
         mesa_loge("panvk: v9 vtc_jc job chain failed: kbase event 0x%x "
                   "(PANVK_V9_TRACE=1 dumps the job status)", -ret);
      if (pan_v9_trace_enabled() || getenv("PANVK_HEAP_LOG")) {
         fprintf(panvk_debug_log_file(),
                 "[PANVK_DEBUG_SUBMIT] vtc_jc ret=%d\n", ret);
         panvk_debug_dump_job_status(cmdbuf, batch, "vtc_jc");
         panvk_debug_dump_tiler_heap(dev, "SESUDAH vtc_jc (vertex+tiler)");
      }

      if (PANVK_DEBUG(TRACE)) {
         panvk_pool_invalidate_maps(&cmdbuf->desc_pool);
      }

      if (PANVK_DEBUG(TRACE)) {
         pandecode_jc(dev->debug.decode_ctx, batch->vtc_jc.first_job,
                      phys_dev->kmod.dev->props.gpu_id);
      }

      if (PANVK_DEBUG(DUMP))
         pandecode_dump_mappings(dev->debug.decode_ctx);

   }

   if (batch->frag_jc.first_job) {
      ret = panvk_kbase_submit_and_wait(dev->drm_fd, batch->frag_jc.first_job, 0x01);
      if (ret)
         mesa_loge("panvk: v9 frag_jc job chain failed: kbase event 0x%x "
                   "(PANVK_V9_TRACE=1 dumps the job status)", -ret);
      if (pan_v9_trace_enabled() || getenv("PANVK_HEAP_LOG")) {
         fprintf(panvk_debug_log_file(),
                 "[PANVK_DEBUG_SUBMIT] frag_jc ret=%d\n", ret);
         panvk_debug_dump_job_status(cmdbuf, batch, "frag_jc");
         panvk_debug_dump_tiler_heap(dev, "SESUDAH frag_jc (fragment)");
      }
      if (PANVK_DEBUG(TRACE)) {
         /* If we want to read the descriptors back, we need to invalidate the
          * whole desc pool, otherwise we might end up with stale data. */
         panvk_pool_invalidate_maps(&cmdbuf->desc_pool);
         pan_kmod_flush_bo_map_syncs(dev->kmod.dev);
      }

      if (PANVK_DEBUG(TRACE))
         pandecode_jc(dev->debug.decode_ctx, batch->frag_jc.first_job,
                      phys_dev->kmod.dev->props.gpu_id);

      if (PANVK_DEBUG(DUMP))
         pandecode_dump_mappings(dev->debug.decode_ctx);

   }

   if (PANVK_DEBUG(TRACE))
      pandecode_next_frame(dev->debug.decode_ctx);

   batch->issued = true;
}

static void
panvk_queue_transfer_sync(struct panvk_gpu_queue *queue, uint32_t syncobj)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   ASSERTED int ret;

   struct drm_syncobj_handle handle = {
      .handle = queue->sync,
      .flags = DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE,
      .fd = -1,
   };

   ret = pan_kmod_ioctl(dev->drm_fd, DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, &handle);
   assert(!ret);
   assert(handle.fd >= 0);

   handle.handle = syncobj;
   ret = pan_kmod_ioctl(dev->drm_fd, DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, &handle);
   assert(!ret);

   close(handle.fd);
}

static void
panvk_add_wait_event_syncobjs(struct panvk_batch *batch, uint32_t *in_fences,
                              unsigned *nr_in_fences)
{
   util_dynarray_foreach(&batch->event_ops, struct panvk_cmd_event_op, op) {
      switch (op->type) {
      case PANVK_EVENT_OP_SET:
         /* Nothing to do yet */
         break;
      case PANVK_EVENT_OP_RESET:
         /* Nothing to do yet */
         break;
      case PANVK_EVENT_OP_WAIT:
         /* kbase: the set happened in an earlier, already finished batch
          * or on the host before submission (required by the spec). */
         if (!op->event->kbase)
            in_fences[(*nr_in_fences)++] = op->event->syncobj;
         break;
      default:
         UNREACHABLE("bad panvk_cmd_event_op type\n");
      }
   }
}

static void
panvk_signal_event_syncobjs(struct panvk_gpu_queue *queue,
                            struct panvk_batch *batch)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);

   util_dynarray_foreach(&batch->event_ops, struct panvk_cmd_event_op, op) {
      switch (op->type) {
      case PANVK_EVENT_OP_SET: {
         if (op->event->kbase) {
            /* The batch has completed (synchronous kbase submit).
             * PANVK_KBASE_EVENT_NO_SET=1: negative control, drop the set. */
            if (!getenv("PANVK_KBASE_EVENT_NO_SET"))
               p_atomic_set(&op->event->kbase_set, 1);
            break;
         }
         panvk_queue_transfer_sync(queue, op->event->syncobj);
         break;
      }
      case PANVK_EVENT_OP_RESET: {
         struct panvk_event *event = op->event;
         if (event->kbase) {
            p_atomic_set(&event->kbase_set, 0);
            break;
         }

         struct drm_syncobj_array objs = {
            .handles = (uint64_t)(uintptr_t)&event->syncobj,
            .count_handles = 1};

         ASSERTED int ret = pan_kmod_ioctl(dev->drm_fd,
                                  DRM_IOCTL_SYNCOBJ_RESET, &objs);
         assert(!ret);
         break;
      }
      case PANVK_EVENT_OP_WAIT:
         /* Nothing left to do */
         break;
      default:
         UNREACHABLE("bad panvk_cmd_event_op type\n");
      }
   }
}

VkResult
panvk_per_arch(gpu_queue_submit)(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct panvk_gpu_queue *queue = container_of(vk_queue, struct panvk_gpu_queue, vk);
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);

   unsigned nr_semaphores = submit->wait_count + 1;
   uint32_t semaphores[nr_semaphores];

   /* On kbase the device's sync type is the CPU sync from
    * get_device_sync_types_kbase(), not a DRM syncobj, so the waits cannot be
    * turned into syncobj handles. Every v9 submission below is synchronous
    * (panvk_kbase_submit_and_wait), so honouring a wait just means waiting for
    * it on the CPU before submitting. */
   semaphores[0] = 0;
   for (unsigned i = 0; i < submit->wait_count; i++) {
      VkResult wr = vk_sync_wait(&dev->vk, submit->waits[i].sync,
                                 submit->waits[i].wait_value,
                                 VK_SYNC_WAIT_COMPLETE, UINT64_MAX);
      if (wr != VK_SUCCESS)
         return wr;
      semaphores[i + 1] = 0;
   }

   /* One chain per submit, shared state: serialise submits on all queues
    * (atom numbers restart at 1 every flush). */
   simple_mtx_lock(&panvk_kbase_chain_mtx);
   struct panvk_kbase_chain *chain =
      panvk_kbase_chain_enabled() ? &panvk_kbase_chain : NULL;
   int cret;

   for (uint32_t j = 0; j < submit->command_buffer_count; ++j) {
      struct panvk_cmd_buffer *cmdbuf =
         container_of(submit->command_buffers[j], struct panvk_cmd_buffer, vk);

      list_for_each_entry(struct panvk_batch, batch, &cmdbuf->batches, node) {
         /* FIXME: should be done at the batch level */
         unsigned nr_bos = panvk_pool_num_bos(&cmdbuf->desc_pool) +
                           panvk_pool_num_bos(&cmdbuf->varying_pool) +
                           panvk_pool_num_bos(&cmdbuf->tls_pool) +
                           batch->fb.bo_count + (batch->blit.src ? 1 : 0) +
                           (batch->blit.dst ? 1 : 0) +
                           (batch->vtc_jc.first_tiler ? 1 : 0) + 1;
         unsigned bo_idx = 0;
         uint32_t bos[nr_bos];

         panvk_pool_get_bo_handles(&cmdbuf->desc_pool, &bos[bo_idx]);
         bo_idx += panvk_pool_num_bos(&cmdbuf->desc_pool);

         panvk_pool_get_bo_handles(&cmdbuf->varying_pool, &bos[bo_idx]);
         bo_idx += panvk_pool_num_bos(&cmdbuf->varying_pool);

         panvk_pool_get_bo_handles(&cmdbuf->tls_pool, &bos[bo_idx]);
         bo_idx += panvk_pool_num_bos(&cmdbuf->tls_pool);

         for (unsigned i = 0; i < batch->fb.bo_count; i++)
            bos[bo_idx++] = pan_kmod_bo_handle(batch->fb.bos[i]);

         if (batch->blit.src)
            bos[bo_idx++] = pan_kmod_bo_handle(batch->blit.src);

         if (batch->blit.dst)
            bos[bo_idx++] = pan_kmod_bo_handle(batch->blit.dst);

         if (batch->vtc_jc.first_tiler)
            bos[bo_idx++] = pan_kmod_bo_handle(dev->tiler_heap->bo);

         bos[bo_idx++] = pan_kmod_bo_handle(dev->sample_positions->bo);
         assert(bo_idx == nr_bos);

         /* Merge identical BO entries. */
         for (unsigned x = 0; x < nr_bos; x++) {
            for (unsigned y = x + 1; y < nr_bos;) {
               if (bos[x] == bos[y])
                  bos[y] = bos[--nr_bos];
               else
                  y++;
            }
         }

         unsigned nr_in_fences = 0;
         unsigned max_wait_event_syncobjs = util_dynarray_num_elements(
            &batch->event_ops, struct panvk_cmd_event_op);
         uint32_t in_fences[nr_semaphores + max_wait_event_syncobjs];
         memcpy(in_fences, semaphores, nr_semaphores * sizeof(*in_fences));
         nr_in_fences += nr_semaphores;

         panvk_add_wait_event_syncobjs(batch, in_fences, &nr_in_fences);

         /* Event ops are applied on the CPU around a batch that has finished:
          * keep such a batch out of a chain (everything before it completes
          * first, and it completes before its sets/resets are applied). */
         const bool has_event_ops = max_wait_event_syncobjs > 0;
         if (chain && has_event_ops &&
             (cret = panvk_kbase_chain_flush(dev->drm_fd, chain)))
            mesa_loge("panvk: v9 chained submit failed: %d", cret);

         panvk_queue_submit_batch(queue, cmdbuf, batch, bos, nr_bos, in_fences,
                                  nr_in_fences, chain);

         if (chain && has_event_ops &&
             (cret = panvk_kbase_chain_flush(dev->drm_fd, chain)))
            mesa_loge("panvk: v9 chained submit failed: %d", cret);

         panvk_signal_event_syncobjs(queue, batch);
      }
   }

   /* Everything queued has to finish before the fences below are signalled. */
   if (chain && (cret = panvk_kbase_chain_flush(dev->drm_fd, chain)))
      mesa_loge("panvk: v9 chained submit failed: %d", cret);
   simple_mtx_unlock(&panvk_kbase_chain_mtx);

   /* Signal fences and semaphores. The old code transferred a DRM syncobj
    * with DRM_IOCTL_SYNCOBJ_* ioctls, which fail on a kbase fd; the asserts
    * are compiled out in release builds, so the failure was silent and no
    * fence was ever signalled: vkWaitForFences always timed out even though
    * the jobs had completed. All work above has already finished, because
    * each batch is submitted and waited for synchronously, so the signal can
    * be raised on the CPU. PANVK_KBASE_NO_SIGNAL=1 reproduces the old
    * behaviour as a negative control. */
   if (!getenv("PANVK_KBASE_NO_SIGNAL")) {
      for (unsigned i = 0; i < submit->signal_count; i++) {
         VkResult sr = vk_sync_signal(&dev->vk, submit->signals[i].sync,
                                      submit->signals[i].signal_value);
         if (sr != VK_SUCCESS)
            return sr;
      }
   }

   return VK_SUCCESS;
}

VkResult
panvk_per_arch(create_gpu_queue)(struct panvk_device *device,
                                 const VkDeviceQueueCreateInfo *create_info,
                                 uint32_t queue_idx,
                                 struct vk_queue **out_queue)
{
   ASSERTED const VkDeviceQueueGlobalPriorityCreateInfoKHR *priority_info =
      vk_find_struct_const(create_info->pNext,
                           DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR);
   ASSERTED const VkQueueGlobalPriorityKHR priority =
      priority_info ? priority_info->globalPriority
                    : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;

   /* XXX: Panfrost kernel module doesn't support priorities so far */
   assert(priority == VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR);

   struct panvk_gpu_queue *queue = vk_zalloc(&device->vk.alloc, sizeof(*queue), 8,
                                         VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!queue)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      vk_queue_init(&queue->vk, &device->vk, create_info, queue_idx);
   if (result != VK_SUCCESS)
      goto err_free_queue;

   queue->sync = 0;

   queue->vk.driver_submit = panvk_per_arch(gpu_queue_submit);
   *out_queue = &queue->vk;
   return VK_SUCCESS;

err_finish_queue:
   vk_queue_finish(&queue->vk);

err_free_queue:
   vk_free(&device->vk.alloc, queue);
   return result;
}

void panvk_per_arch(destroy_gpu_queue)(struct vk_queue *vk_queue)
{
   struct panvk_gpu_queue *queue = container_of(vk_queue, struct panvk_gpu_queue, vk);
   struct panvk_device *dev = to_panvk_device(vk_queue->base.device);

   vk_queue_finish(&queue->vk);
   /* PATCH: no drm_syncobj to destroy */
   vk_free(&dev->vk.alloc, queue);
}

VkResult
panvk_per_arch(gpu_queue_check_status)(struct vk_queue *vk_queue)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(QueueWaitIdle)(VkQueue _queue)
{
   VK_FROM_HANDLE(panvk_gpu_queue, queue, _queue);
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);

   /* we need to use vk_common_QueueWaitIdle if we ever go threaded */
   assert(queue->vk.submit.mode != VK_QUEUE_SUBMIT_MODE_THREADED);

   if (vk_device_is_lost(&dev->vk)) {
      /* Check printf buffer one more time before exiting */
      u_printf_with_ctx(stdout, &dev->printf.ctx);
      return VK_ERROR_DEVICE_LOST;
   }

   return VK_SUCCESS;
}
