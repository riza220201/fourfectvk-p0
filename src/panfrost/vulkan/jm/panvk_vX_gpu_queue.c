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

#include "c11/threads.h"
#include "util/u_atomic.h"
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
#include <sys/eventfd.h>
#include <unistd.h>
#include "util/cnd_monotonic.h"
#include "util/libsync.h"
#include "util/os_time.h"
#include "util/timespec.h"
#include "pan_v9_trace.h"
#include "pan_props.h"
#include "pan_desc.h"
#include "panfrost/lib/pan_env.h"
#include "panvk_mem_prof.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_PAN_KMOD_KBASE
#include "kmod/kbase_kmod.h"
#else
static inline void
kbase_kmod_describe_va(uint64_t va, char *buf, size_t size)
{
   snprintf(buf, size, "?");
}
static inline int
kbase_kmod_va_check(uint64_t va, uint64_t len, void *out, bool *growable,
                    uint64_t *bo_va, uint64_t *bo_size)
{
   return -3;
}
struct kbase_kmod_mem_stats {
   uint64_t growable_commit_bytes;
};
static inline void
kbase_kmod_mem_stats(struct kbase_kmod_mem_stats *out)
{
}
static inline void
kbase_kmod_quarantine_hold(struct pan_kmod_dev *dev)
{
}
static inline unsigned
kbase_kmod_quarantine_release(struct pan_kmod_dev *dev, uint64_t *bytes_out)
{
   return 0;
}
#endif

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
static once_flag panvk_debug_log_once = ONCE_FLAG_INIT;
static FILE *panvk_debug_log_stream;

static void
panvk_debug_log_init(void)
{
   const char *path = getenv("PANVK_HEAP_LOG");

   if (path && path[0]) {
      panvk_debug_log_stream = fopen(path, "ae");
      if (panvk_debug_log_stream) {
         setvbuf(panvk_debug_log_stream, NULL, _IOLBF, 0);
         fprintf(panvk_debug_log_stream, "[PANVK_DEBUG_HEAP] --- log dibuka, pid=%d ---\n",
                 (int)getpid());
      } else {
         fprintf(stderr,
                 "[PANVK_DEBUG_HEAP] gagal membuka PANVK_HEAP_LOG=%s: %s,"
                 " jatuh ke stderr\n",
                 path, strerror(errno));
      }
   }
}

static FILE *
panvk_debug_log_file(void)
{
   call_once(&panvk_debug_log_once, panvk_debug_log_init);
   return panvk_debug_log_stream ? panvk_debug_log_stream : stderr;
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
   case 0x42:
      return "JOB_READ_FAULT";
   case 0x4002:
      return "JOB_CANCELLED";
   case 0x4003:
      return "JOB_INVALID";
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

/* ---- kbase JM submission: kernel-side chaining (8.9), async completion (8.15)
 *
 * 8.9: the atoms of a vkQueueSubmit are queued with a DATA dependency on the
 * previous atom -- the order of the old submit-and-wait loop, enforced by
 * kbase -- and go to the kernel in one JOB_SUBMIT. Only the last atom reports
 * success; a failed atom reports itself and cancels its dependents.
 *
 * 8.15: vkQueueSubmit no longer waits for its atoms. Each JOB_SUBMIT is a
 * submission with a seqno; whichever thread needs progress (a CPU wait, a
 * status query, an allocation short of atom numbers) reads the kbase
 * completion events, one reader at a time, and advances completed_seqno (a
 * dedicated event thread cost ~0.1-0.25 ms per wait in two wake-ups,
 * measured). A fence or semaphore signalled by a
 * submit is pending on its seqno: a CPU wait blocks on completed_seqno, and a
 * SYNC_FD export submits a FENCE_TRIGGER soft atom that depends on the
 * submission's last atom -- a real sync_file, signalled by kbase. A waited-on
 * imported sync_file becomes a FENCE_WAIT soft atom that the first job atom
 * depends on, so the GPU waits for it instead of the CPU. Every submission's
 * first atom depends (ORDER) on the previous submission's last atom: work on
 * the queue still runs in submission order, a wait on our own earlier work
 * needs nothing, and the shared tiler heap stays serialised.
 *
 * Kernel behaviour relied on (a16w_jm r54p1 source, and fenceprobe on the
 * RS4): an atom number can be reused once its event has been read, or at once
 * if the event was suppressed on success; a dependency on a completed atom is
 * met if that atom ended DONE; an ORDER dependency does not pass a failure on;
 * the FENCE_TRIGGER fd is written back during the JOB_SUBMIT ioctl.
 *
 * PANVK_KBASE_WAIT_SUBMIT=1 restores the 8.14 behaviour: waits on the CPU
 * before, the submit waits for its atoms, signals on the CPU, exports as a CPU
 * wait + -1. PANVK_KBASE_CPU_FENCE_WAIT=1 waits for imported sync_files on
 * the CPU; PANVK_KBASE_CPU_FENCE_EXPORT=1 exports as a CPU wait + -1.
 * PANVK_KBASE_SYNC_SUBMIT=1 (8.9's per-chain control) implies WAIT_SUBMIT.
 * Negative controls, broken on purpose: PANVK_KBASE_CTRL_NO_FENCE_WAIT_DEP=1
 * (work does not wait for imported sync_files), PANVK_KBASE_CTRL_EXPORT_EARLY=1
 * (an export claims the work is done: -1, without waiting). */
#define PANVK_KBASE_MAX_CHAIN 120
#define PANVK_KBASE_ATOM_STRIDE 64 /* base_jd_atom_v2 + MTK BM_2 frame_nr */
/* The job atoms of one submission, plus a FENCE_WAIT and a DEP atom. */
#define PANVK_KBASE_MAX_SUB_ATOMS (PANVK_KBASE_MAX_CHAIN + 2)
/* Each submission in flight holds at least one of the 255 atom numbers. */
#define PANVK_KBASE_MAX_SUBS 255
/* From the kbase JM UAPI, mali_base_jm_kernel.h (r54p1). */
#define BASE_JD_DEP_TYPE_DATA (1U << 0)
#define BASE_JD_DEP_TYPE_ORDER (1U << 1)
#define BASE_JD_REQ_DEP 0U
#define BASE_JD_REQ_PERMON (1U << 7)
#define BASE_JD_REQ_SOFT_JOB (1U << 9)
#define BASE_JD_REQ_SOFT_FENCE_TRIGGER (BASE_JD_REQ_SOFT_JOB | 0x2)
#define BASE_JD_REQ_SOFT_FENCE_WAIT (BASE_JD_REQ_SOFT_JOB | 0x3)
#define BASE_JD_REQ_EVENT_ONLY_ON_FAILURE (1U << 12)
#define BASE_JD_EVENT_DONE 0x1
#define BASE_JD_EVENT_JOB_CANCELLED 0x4002
#define BASE_JD_REQ_CS (1U << 1)

/* P0 (FourFectVK T06/T08): after a failed atom the Vulkan device is lost,
 * but the GPU may still be running the failed work -- a hung shader until
 * kbase hard-stops it (~5 s) or resets the GPU (~8.5 s), other atoms until
 * they report -- and the application now frees everything. Measured on the
 * RS4 (mesa-mobile JOURNAL (ai)): kbase posts an event for every atom, but
 * neither the event nor an exported fence proves the GPU has stopped (a
 * FENCE_TRIGGER can signal during the GPU reset), and closing the fd does
 * not order the frees after the jobs (mappings keep the context alive until
 * they are gone). What does prove it: a new GPU job completing after every
 * failed atom has reported -- kbase starts nothing on the GPU while a reset
 * is pending (kbase_backend_slot_update), so its DONE comes after any reset.
 * So from the loss on, freed BOs are held mapped (kbase_kmod_quarantine_*),
 * a settle thread waits for every in-flight atom to report and for such a
 * barrier job, then releases them. Work that never ends (a foreign fence
 * that is never signalled, a job chain that obeys soft-stop forever) cannot
 * settle: the memory stays held until the device is destroyed; neither case
 * has the GPU under reset, so freeing then faults on the GPU instead of
 * reaching freed pages (measured). PANVK_KBASE_P0=0 turns all of it off
 * (control); PANVK_KBASE_P0_DEADLINE_MS (12000) bounds the settle. */
#define PANVK_KBASE_P0_ATOM 255 /* reserved for the settle barrier */
#define PANVK_KBASE_P0_SLOT_SIZE 128 /* one WRITE_VALUE job per atom number */

/* struct base_fence, mali_base_kernel.h */
struct panvk_kbase_base_fence {
   int fd;
   int stream_fd;
};

struct panvk_kbase_job {
   uint64_t jc;
   uint32_t core_req;
   bool overlap_second;
   /* PANVK_FAULT_REPORT: what the atom carries (0076). */
   struct panvk_cmd_buffer *cmdbuf;
   struct panvk_batch *batch;
   unsigned batch_idx;
};

/* PANVK_FAULT_REPORT: what each atom number carries while in flight. */
struct panvk_kbase_atom_info {
   uintptr_t cmdbuf_id;
   uint64_t jc;
   uint32_t core_req;
   unsigned batch_idx;
   unsigned nr_jobs;
};

static struct panvk_kbase_atom_info
panvk_kbase_job_info(const struct panvk_kbase_job *job, bool snapshot)
{
   return (struct panvk_kbase_atom_info){
      .cmdbuf_id = (uintptr_t)job->cmdbuf,
      .jc = job->jc,
      .core_req = job->core_req,
      .batch_idx = job->batch_idx,
      .nr_jobs = snapshot && job->batch
                    ? util_dynarray_num_elements(&job->batch->jobs, void *)
                    : 0,
   };
}

/* The job atoms of the vkQueueSubmit being built, not yet in the kernel. */
struct panvk_kbase_chain {
   struct panvk_kbase_job jobs[PANVK_KBASE_MAX_CHAIN];
   struct panvk_batch *batches[PANVK_KBASE_MAX_CHAIN];
   unsigned nr_jobs, nr_batches;
   /* Merged imported sync_files the next submission waits for, or -1. */
   int wait_fd;
};

struct panvk_kbase_sub {
   uint64_t seqno;
   uint8_t last; /* the atom whose event completes the submission */
   bool done;
   bool terminal; /* P0: its last atom reported, whatever the code */
   int fence_fd; /* out-fence made by an export, -1 until then */
   uint64_t submit_ns; /* PANVK_FRAME_PROF */
   uint16_t nr_atoms;
   uint8_t atoms[PANVK_KBASE_MAX_SUB_ATOMS];
};

enum panvk_kbase_atom_use {
   PANVK_KBASE_ATOM_FREE = 0,
   PANVK_KBASE_ATOM_SUB, /* in a submission, freed when it completes */
   PANVK_KBASE_ATOM_AUX, /* an export's FENCE_TRIGGER, freed by its event */
};

struct panvk_kbase_jm {
   int fd;
   int wake_fd;
   bool closing;
   struct panvk_kbase_owner *owner;
   /* Protects everything below, and is held across JOB_SUBMIT so that seqno
    * order is kernel submission order. The reader polls and reads without
    * it. */
   mtx_t mtx;
   struct u_cnd_monotonic cnd; /* completions processed, or reader done */
   uint64_t last_seqno;
   uint64_t completed_seqno;
   struct panvk_kbase_sub subs[PANVK_KBASE_MAX_SUBS]; /* ring, oldest first */
   unsigned head, count;
   uint8_t atom_use[256];
   unsigned nr_free, cursor;
   bool lost;
   bool reading; /* a thread is polling/reading the kbase fd */
   /* Only touched by the submitting thread (queue access is synchronised). */
   struct panvk_kbase_chain chain;
   bool wait_submit, cpu_fence_wait, cpu_fence_export;
   bool ctrl_no_fence_wait_dep, ctrl_export_early;
   /* Read once at creation (were read on every submit). */
   bool chain_enabled, no_signal;
   bool overlap_enabled;
   struct {
      struct panvk_priv_bo *heap, *tls, *wls;
   } overlap_banks[2];
   /* Counted always, printed at destroy with PANVK_KBASE_FENCE_TRACE=1:
    * waits by payload, FENCE_WAIT atoms, exports by outcome. */
   bool trace;
   struct {
      uint64_t wait_none, wait_own, wait_file, wait_cpu, fence_wait_atoms;
      uint64_t export_new, export_dup, export_done, export_cpu;
   } stats;

   /* FourFectVK diagnostics (0076/0077), all under mtx. */
   struct panvk_device *dev;
   uint64_t t0_ns, fail_print_ns;
   unsigned errors;
   bool fault_report, audit_all;
   unsigned fault_reports;
   unsigned audit_batches, audit_suspect_batches, audit_lines;
   struct panvk_kbase_atom_info atom_info[256];
   /* PANVK_FRAME_PROF: end of the GPU busy time counted so far. */
   uint64_t busy_end_ns;

   /* P0, see PANVK_KBASE_P0_ATOM. Under mtx except where noted. */
   struct {
      bool enabled;
      bool started;      /* loss latched: quarantine held, settle started */
      bool thread_ok;
      bool settled;      /* proven: quarantine released */
      bool barrier_seen;
      uint32_t barrier_code;
      thrd_t thread;
      uint64_t lost_ns, deadline_ns;
      /* The last kbase event before the loss: kbase bounds GPU work (hard-
       * stop, reset) from there, so the settle deadline counts from it. A
       * watchdog loss comes long after it: no point waiting again. */
      uint64_t progress_at_loss_ns;
      unsigned aux_inflight; /* export FENCE_TRIGGERs not yet reported */
      struct panvk_priv_bo *bo; /* barrier WRITE_VALUE jobs, one per atom */
   } p0;
   /* An export's barrier atom, freed with its FENCE_TRIGGER (0 = none). */
   uint8_t aux_pair[256];
   bool export_barrier;
   /* Watchdog: a wait with work in flight and no event for watchdog_ns
    * latches a loss (PANVK_KBASE_WATCHDOG_MS, 0 = off). */
   uint64_t watchdog_ns, last_progress_ns;
};

static void panvk_kbase_mark_lost(struct panvk_kbase_jm *k);

static bool
panvk_kbase_chain_enabled(void)
{
   /* The trace/dump paths decode each chain right after it ran. */
   return !pan_env_flag("PANVK_KBASE_SYNC_SUBMIT") && !pan_v9_trace_enabled() &&
          !getenv("PANVK_HEAP_LOG") && !PANVK_DEBUG(TRACE) && !PANVK_DEBUG(DUMP);
}

static void
panvk_kbase_atom(uint8_t *a, uint8_t num, uint64_t jc, uint32_t core_req,
                 uint8_t dep0, uint8_t type0, uint8_t dep1, uint8_t type1)
{
   memset(a, 0, PANVK_KBASE_ATOM_STRIDE);
   memcpy(a + 0, &jc, 8);
   a[36] = dep0;                /* pre_dep[0].atom_id */
   a[37] = dep0 ? type0 : 0;    /* pre_dep[0].dependency_type */
   a[38] = dep1;                /* pre_dep[1] */
   a[39] = dep1 ? type1 : 0;
   a[40] = num;                 /* atom_number */
   memcpy(a + 44, &core_req, 4);
}

static void
panvk_kbase_free_atoms(struct panvk_kbase_jm *k, unsigned n,
                       const uint8_t *nums)
{
   for (unsigned i = 0; i < n; i++) {
      assert(k->atom_use[nums[i]] != PANVK_KBASE_ATOM_FREE);
      k->atom_use[nums[i]] = PANVK_KBASE_ATOM_FREE;
   }
   k->nr_free += n;
}

static struct panvk_kbase_sub *
panvk_kbase_find_sub(struct panvk_kbase_jm *k, uint8_t atom)
{
   for (unsigned i = 0; i < k->count; i++) {
      struct panvk_kbase_sub *sub =
         &k->subs[(k->head + i) % PANVK_KBASE_MAX_SUBS];
      for (unsigned j = 0; j < sub->nr_atoms; j++) {
         if (sub->atoms[j] == atom)
            return sub;
      }
   }
   return NULL;
}

/* PATCH: PANVK_FAULT_REPORT=1. Prints, for a failed atom, the batch it
 * carried: which chain, the job types, the status the GPU wrote back into
 * each job header and the fault pointer, plus what the fault pointer is
 * (live GPU memory, a quarantined freed BO with PANVK_DBG_FREE_DELAY_MS,
 * or nothing). The command buffer is still pending, so its descriptors are
 * alive. Caller holds k->mtx. */
static const char *
panvk_job_type_str(unsigned t)
{
   switch (t) {
   case 0: return "not-started";
   case 1: return "null";
   case 2: return "write-value";
   case 3: return "cache-flush";
   case 4: return "compute";
   case 7: return "tiler";
   case 9: return "fragment";
   case 10: return "indexed-vertex";
   case 11: return "malloc-vertex";
   case 12: return "fullscreen";
   default: return "?";
   }
}

/* PATCH 0077: pointer audit of a failed batch. kbase kills the context's
 * jobs on an MMU fault without telling userspace the address (dmesg only),
 * and a killed job never writes its status, so the job headers say
 * nothing. Instead, every GPU address the failed chain follows (job
 * descriptors -> draw/shader descriptors -> resource tables -> buffer,
 * texture and plane descriptors -> memory; framebuffer -> render targets,
 * tiler heap) is checked against the live BOs: an address in no BO, in a
 * freed BO or running past the end of its BO is printed, and the growable
 * (grow-on-fault) BOs that are used are listed with how much the kernel has
 * backed. Descriptor memory is read through the BO's CPU mapping; growable
 * BOs are never read. */
struct fault_audit {
   unsigned checked, outside, freed, past_end, sink, shown;
   unsigned show_max;
   unsigned occl_len, jobs_skipped;
   unsigned nseen;
   uint64_t seen[256];
   unsigned ngrow;
   struct {
      uint64_t bo, size, max_end;
      unsigned refs;
   } grow[16];
};

static uint32_t
fa_rd32(const uint8_t *p)
{
   uint32_t v;
   memcpy(&v, p, 4);
   return v;
}

static uint64_t
fa_rd64(const uint8_t *p)
{
   uint64_t v;
   memcpy(&v, p, 8);
   return v;
}

/* True the first time a table address is seen (tables are shared by
 * many draws). */
static bool
fa_first(struct fault_audit *a, uint64_t va)
{
   for (unsigned i = 0; i < a->nseen; i++)
      if (a->seen[i] == va)
         return false;
   if (a->nseen < ARRAY_SIZE(a->seen))
      a->seen[a->nseen++] = va;
   return true;
}

static void
fa_ptr(struct fault_audit *a, unsigned job, const char *what, uint64_t va,
       uint64_t len)
{
   if (!va)
      return;
   a->checked++;
   if ((va >> 60) == 0x8) {
      /* Memory sink of null dynamic buffers: never dereferenced as long
       * as the size is 0. */
      a->sink++;
      if (!len)
         return;
   }

   bool growable = false;
   uint64_t bo = 0, bo_size = 0;
   int r = kbase_kmod_va_check(va, len ? len : 1, NULL, &growable, &bo,
                               &bo_size);
   if (r == 0) {
      if (growable) {
         unsigned i;
         for (i = 0; i < a->ngrow; i++)
            if (a->grow[i].bo == bo)
               break;
         if (i == a->ngrow && a->ngrow < ARRAY_SIZE(a->grow)) {
            a->grow[i].bo = bo;
            a->grow[i].size = bo_size;
            a->grow[i].max_end = 0;
            a->grow[i].refs = 0;
            a->ngrow++;
         }
         if (i < a->ngrow) {
            a->grow[i].refs++;
            a->grow[i].max_end =
               MAX2(a->grow[i].max_end, va - bo + (len ? len : 1));
         }
      }
      return;
   }
   if (r == -3)
      return;

   if (r == -1)
      a->freed++;
   else if (r == -4)
      a->past_end++;
   else
      a->outside++;

   if (a->shown++ < a->show_max) {
      char where[160];
      kbase_kmod_describe_va(va, where, sizeof(where));
      mesa_loge("panvk-fault:   SUSPECT job[%u] %s 0x%" PRIx64 " +0x%" PRIx64
                ": %s%s", job, what, va, len, where,
                r == -4 ? " (runs past the end of the BO)" : "");
   }
}

static bool
fa_read(uint64_t va, void *out, size_t len)
{
   return va && (va >> 60) != 0x8 &&
          kbase_kmod_va_check(va, len, out, NULL, NULL, NULL) == 0;
}

static void
fa_desc(struct fault_audit *a, unsigned job, const uint8_t *d)
{
   switch (d[0] & 0xf) {
   case 2: { /* Texture */
      unsigned levels = ((fa_rd32(d + 8) >> 16) & 0x1f) + 1;
      unsigned layers = (fa_rd32(d + 24) & 0xffff) + 1;
      uint64_t surfaces = fa_rd64(d + 16);
      uint64_t n = MIN2((uint64_t)levels * layers, 4096);
      fa_ptr(a, job, "texture surfaces", surfaces, n * 32);
      uint8_t pl[32];
      for (uint64_t i = 0; i < n; i += (n > 1 ? n - 1 : 1)) {
         if (!fa_read(surfaces + i * 32, pl, 32))
            break;
         fa_ptr(a, job, "texture plane", fa_rd64(pl + 8), fa_rd32(pl + 4));
      }
      break;
   }
   case 8: /* Shader program */
      fa_ptr(a, job, "shader binary", fa_rd64(d + 8), 4);
      break;
   case 9: /* Buffer */
      fa_ptr(a, job, "buffer", fa_rd64(d + 8), fa_rd32(d + 4));
      break;
   case 10: /* Plane */
      fa_ptr(a, job, "plane", fa_rd64(d + 8), fa_rd32(d + 4));
      break;
   default:
      break;
   }
}

static void
fa_res_table(struct fault_audit *a, unsigned job, uint64_t res)
{
   const uint64_t tab = res & ~0x3full;
   const unsigned count = res & 0x3f;

   if (!tab || !fa_first(a, res))
      return;
   fa_ptr(a, job, "resource table", tab, count * 16);
   for (unsigned i = 0; i < count; i++) {
      uint8_t e[16];
      if (!fa_read(tab + i * 16, e, 16))
         continue;
      uint64_t addr = fa_rd64(e) & BITFIELD64_MASK(56);
      uint64_t size = fa_rd64(e + 8);
      if (!addr || !size || !fa_first(a, addr))
         continue;
      fa_ptr(a, job, "descriptor table", addr, size);
      uint64_t n = MIN2(size / 32, 1024);
      for (uint64_t k = 0; k < n; k++) {
         uint8_t d[32];
         if (!fa_read(addr + k * 32, d, 32))
            break;
         fa_desc(a, job, d);
      }
   }
}

/* Shader Environment, 64 bytes. Returns the shader binary address. */
static uint64_t
fa_shader_env(struct fault_audit *a, unsigned job, const char *what,
              const uint8_t *env)
{
   uint64_t prog = fa_rd64(env + 40), tls = fa_rd64(env + 48);
   uint64_t fau = fa_rd64(env + 56), binary = 0;
   unsigned fau_count = fa_rd32(env + 4) & 0xff;
   char w[48];

   snprintf(w, sizeof(w), "%s shader program", what);
   fa_ptr(a, job, w, prog, 32);
   uint8_t sp[32];
   if (fa_read(prog, sp, 32)) {
      binary = fa_rd64(sp + 8);
      snprintf(w, sizeof(w), "%s shader binary", what);
      fa_ptr(a, job, w, binary, 4);
   }
   if (tls && fa_first(a, tls)) {
      fa_ptr(a, job, "local storage", tls, 32);
      uint8_t ls[32];
      if (fa_read(tls, ls, 32)) {
         /* Packed address mode (what pan_emit_tls uses on v9) stores the
          * base shifted right by 8. */
         uint64_t tls_base = fa_rd64(ls + 8) & BITFIELD64_MASK(48);
         if ((fa_rd32(ls + 12) >> 28) == 1)
            tls_base <<= 8;
         fa_ptr(a, job, "TLS", tls_base, 1);
         fa_ptr(a, job, "WLS", fa_rd64(ls + 16), 1);
      }
   }
   snprintf(w, sizeof(w), "%s FAU", what);
   fa_ptr(a, job, w, fau, fau_count * 8);
   fa_res_table(a, job, fa_rd64(env + 32));
   return binary;
}

/* Draw descriptor (DCD), 128 bytes: fragment side of a draw. */
static void
fa_draw(struct fault_audit *a, unsigned job, const uint8_t *draw)
{
   uint64_t binary = fa_shader_env(a, job, "fragment", draw + 64);
   uint64_t blend = fa_rd64(draw + 48);
   unsigned blend_count = blend & 0xf;

   blend &= ~0xfull;
   fa_ptr(a, job, "depth/stencil descriptor", fa_rd64(draw + 40), 32);
   /* One 64-bit counter per shader core ID (query pool layout). */
   fa_ptr(a, job, "occlusion", fa_rd64(draw + 56), a->occl_len);
   fa_ptr(a, job, "blend descriptors", blend, blend_count * 16);
   for (unsigned i = 0; i < blend_count; i++) {
      uint8_t b[16];
      if (!fa_read(blend + i * 16, b, 16))
         break;
      uint32_t pc = fa_rd32(b + 12) & ~0xfu;
      if ((fa_rd32(b + 8) & 3) == 0 && pc && binary)
         fa_ptr(a, job, "blend shader", (binary & ~0xffffffffull) | pc, 4);
   }
}

static void
fa_framebuffer(struct fault_audit *a, unsigned job, const uint8_t *frag_job)
{
   uint64_t fbd = fa_rd64(frag_job + 40) & ~0x3full;
   uint8_t p[128];

   fa_ptr(a, job, "framebuffer descriptor", fbd, 128);
   if (!fa_read(fbd, p, 128))
      return;

   unsigned rt_count = ((fa_rd32(p + 44) >> 19) & 0xf) + 1;
   bool has_zs = (fa_rd32(p + 48) >> 21) & 1;
   uint64_t dcds = fa_rd64(p + 24), tiler = fa_rd64(p + 56);
   uint32_t modes = fa_rd32(p);

   fa_ptr(a, job, "sample locations", fa_rd64(p + 16), 8);
   fa_ptr(a, job, "tile enable map", fa_rd64(frag_job + 48), 1);
   for (unsigned i = 0; i < 3; i++) {
      uint8_t dcd[128];
      if (((modes >> (3 * i)) & 7) && fa_read(dcds + i * 128, dcd, 128))
         fa_draw(a, job, dcd);
   }

   /* The tiler updates the polygon list pointer (flags in the top bits)
    * and the heap's bottom/top while it runs: only the 48-bit address and
    * the CPU-written base + size are checked. */
   uint8_t t[32];
   fa_ptr(a, job, "tiler context", tiler, 32);
   if (fa_read(tiler, t, 32)) {
      fa_ptr(a, job, "polygon list", fa_rd64(t) & BITFIELD64_MASK(48), 1);
      uint64_t heap = fa_rd64(t + 24);
      uint8_t h[32];
      fa_ptr(a, job, "tiler heap descriptor", heap, 32);
      if (fa_read(heap, h, 32))
         fa_ptr(a, job, "tiler heap", fa_rd64(h + 8), fa_rd32(h + 4));
   }

   uint64_t rts = fbd + 128;
   if (has_zs) {
      uint8_t z[64];
      fa_ptr(a, job, "ZS/CRC extension", rts, 64);
      if (fa_read(rts, z, 64)) {
         fa_ptr(a, job, "depth target", fa_rd64(z + 32), 1);
         fa_ptr(a, job, "stencil target", fa_rd64(z + 48), 1);
      }
      rts += 64;
   }
   for (unsigned i = 0; i < rt_count; i++) {
      uint8_t rt[64];
      if (!fa_read(rts + i * 64, rt, 64))
         break;
      fa_ptr(a, job, "render target", fa_rd64(rt + 32), 1);
   }
}

/* quiet: PANVK_FAULT_AUDIT_ALL, prints SUSPECT lines only (show_max of
 * them). Returns the number of suspect addresses. */
static unsigned
panvk_fault_audit(struct panvk_kbase_jm *k, struct panvk_batch *batch,
                  bool frag, bool quiet, unsigned show_max)
{
   struct panvk_physical_device *phys_dev =
      to_panvk_physical_device(k->dev->vk.physical);
   struct fault_audit *a = calloc(1, sizeof(*a));
   if (!a)
      return 0;
   a->show_max = show_max;
   a->occl_len = 8 * pan_query_core_id_range(&phys_dev->kmod.dev->props);

   unsigned idx = 0;
   util_dynarray_foreach(&batch->jobs, void *, job) {
      const uint8_t *j = *job;
      unsigned type = (fa_rd32(j + 16) >> 1) & BITFIELD_MASK(7);

      /* Bounded: this runs with k->mtx held. */
      if (idx >= 512) {
         a->jobs_skipped++;
         idx++;
         continue;
      }

      if (type == 11) { /* malloc vertex */
         fa_draw(a, idx, j + 128);
         if (!frag) {
            fa_shader_env(a, idx, "position", j + 256);
            fa_shader_env(a, idx, "varying", j + 320);
         }
      } else if (type == 9 && frag) {
         fa_framebuffer(a, idx, j);
      } else if (type == 4 && !frag) {
         fa_shader_env(a, idx, "compute", j + 64);
      }
      idx++;
   }

   const unsigned suspects = a->outside + a->freed + a->past_end;
   if (quiet) {
      free(a);
      return suspects;
   }

   mesa_loge("panvk-fault:   audit: %u GPU addresses checked: %u in no BO, "
             "%u in freed BOs, %u past the end of their BO, %u null-buffer "
             "sink%s", a->checked, a->outside, a->freed, a->past_end, a->sink,
             a->jobs_skipped ? " (first 512 jobs only)" : "");
   for (unsigned i = 0; i < a->ngrow; i++) {
      char where[160];
      kbase_kmod_describe_va(a->grow[i].bo + a->grow[i].max_end - 1, where,
                             sizeof(where));
      mesa_loge("panvk-fault:   growable BO 0x%" PRIx64 " +0x%" PRIx64
                ": %u refs up to offset 0x%" PRIx64 " (%s)", a->grow[i].bo,
                a->grow[i].size, a->grow[i].refs, a->grow[i].max_end, where);
   }
   free(a);
   return suspects;
}

static uint64_t
fa_mem_available_mb(void)
{
   FILE *f = fopen("/proc/meminfo", "r");
   char line[128];
   uint64_t kb = 0;

   if (!f)
      return 0;
   while (fgets(line, sizeof(line), f)) {
      if (sscanf(line, "MemAvailable: %" SCNu64, &kb) == 1)
         break;
   }
   fclose(f);
   return kb / 1024;
}

static void
panvk_kbase_fault_report(struct panvk_kbase_jm *k, uint8_t atom,
                         uint64_t seqno, uint32_t event_code)
{
   const struct panvk_kbase_atom_info *ai = &k->atom_info[atom];
   const bool frag = ai->core_req == 0x01;

   mesa_loge("panvk-fault: event 0x%x (%s) atom %u submission %" PRIu64
             " %s chain 0x%" PRIx64 ", cmdbuf-id 0x%" PRIxPTR " batch %u",
             event_code,
             panvk_jd_event_str(event_code), atom, seqno,
             !ai->jc ? "no job" : frag ? "fragment" : "vertex/tiler/compute",
             ai->jc, ai->cmdbuf_id, ai->batch_idx);
   mesa_loge("panvk-fault:   submit snapshot: %u batch jobs; "
             "live job status unavailable", ai->nr_jobs);

   struct kbase_kmod_mem_stats ms = {0};
   kbase_kmod_mem_stats(&ms);
   mesa_loge("panvk-fault:   t=%.1fs since device creation, MemAvailable %"
             PRIu64 " MB, growable BOs backed %" PRIu64 " MB",
             (os_time_get_nano() - k->t0_ns) / 1e9,
             fa_mem_available_mb(), ms.growable_commit_bytes >> 20);
}

/* A failed atom. kbase kills every job of the context after an MMU fault,
 * so one fault is followed by many failures: the first 16 are printed, then
 * one summary line per second, so a log shows whether they keep coming
 * without slowing the game down (0077). Caller holds k->mtx. */
static void
panvk_kbase_report_failure(struct panvk_kbase_jm *k, uint8_t atom,
                           uint64_t seqno, uint32_t event_code)
{
   const uint64_t now = os_time_get_nano();

   if (k->errors >= 16 && now - k->fail_print_ns >= 1000000000ull) {
      k->fail_print_ns = now;
      mesa_loge("panvk: %u GPU job chains failed so far (t=%.1fs, last "
                "event 0x%x submission %" PRIu64 ")", k->errors + 1,
                (now - k->t0_ns) / 1e9, event_code, seqno);
   }
   if (k->errors++ < 16)
      mesa_loge("panvk: kbase atom %u of submission %" PRIu64
                " failed: event 0x%x (%s; PANVK_FAULT_REPORT=1 shows what it "
                "carried)", atom, seqno, event_code,
                panvk_jd_event_str(event_code));
   if (k->fault_report && k->fault_reports++ < 64)
      panvk_kbase_fault_report(k, atom, seqno, event_code);
}

/* Caller holds k->mtx. */
static void
panvk_kbase_handle_event(struct panvk_kbase_jm *k,
                         const struct panvk_kbase_event_v2 *ev)
{
   const uint8_t atom = ev->atom_number;

   k->last_progress_ns = os_time_get_nano();

   if (atom == PANVK_KBASE_P0_ATOM && k->p0.bo) {
      k->p0.barrier_seen = true;
      k->p0.barrier_code = ev->event_code;
      return;
   }

   if (k->atom_use[atom] == PANVK_KBASE_ATOM_AUX) {
      /* An export: its FENCE_TRIGGER always reports; its barrier (if any)
       * only on failure, and is freed with the trigger. */
      const bool trigger =
         k->atom_info[atom].core_req == BASE_JD_REQ_SOFT_FENCE_TRIGGER;
      if (trigger && k->p0.aux_inflight)
         k->p0.aux_inflight--;
      if (ev->event_code != BASE_JD_EVENT_DONE) {
         mesa_loge("panvk: kbase out-fence atom %u: event 0x%x", atom,
                   ev->event_code);
         panvk_kbase_mark_lost(k);
         vk_device_set_lost(&k->dev->vk, "kbase out-fence failed: 0x%x",
                            ev->event_code);
      }
      if (!k->lost && trigger) {
         panvk_kbase_free_atoms(k, 1, &atom);
         if (k->aux_pair[atom]) {
            const uint8_t barrier = k->aux_pair[atom];
            k->aux_pair[atom] = 0;
            panvk_kbase_free_atoms(k, 1, &barrier);
         }
      }
      return;
   }

   struct panvk_kbase_sub *sub = panvk_kbase_find_sub(k, atom);
   if (!sub) {
      mesa_loge("panvk: kbase event 0x%x for unknown atom %u", ev->event_code,
                atom);
      return;
   }

   if (ev->event_code != BASE_JD_EVENT_DONE &&
       !(k->atom_info[atom].core_req == BASE_JD_REQ_SOFT_FENCE_WAIT &&
         ev->event_code == BASE_JD_EVENT_JOB_CANCELLED)) {
      panvk_kbase_report_failure(k, atom, sub->seqno, ev->event_code);
      panvk_kbase_mark_lost(k);
      vk_device_set_lost(&k->dev->vk, "kbase atom %u failed: 0x%x", atom,
                         ev->event_code);
   }
   /* P0: the last atom depends on every other atom of its submission (DATA
    * chain, the wait by ORDER), so once it reported, all of it has. */
   if (atom == sub->last)
      sub->terminal = true;
   if (k->lost)
      return;

   if (atom == sub->last)
      sub->done = true;

   /* Submissions complete in order: retire the finished ones. */
   const uint64_t now = k->dev->frame_prof.enabled ? os_time_get_nano() : 0;
   while (k->count && k->subs[k->head].done) {
      struct panvk_kbase_sub *s = &k->subs[k->head];
      if (now) {
         /* GPU busy = union of [submit, completion seen] (FRAME_PROF). */
         const uint64_t start = MAX2(s->submit_ns, k->busy_end_ns);
         if (s->submit_ns && now > start)
            p_atomic_add(&k->dev->frame_prof.gpu_busy_ns, now - start);
         k->busy_end_ns = MAX2(k->busy_end_ns, now);
      }
      panvk_kbase_free_atoms(k, s->nr_atoms, s->atoms);
      if (s->fence_fd >= 0)
         close(s->fence_fd);
      k->completed_seqno = s->seqno;
      k->head = (k->head + 1) % PANVK_KBASE_MAX_SUBS;
      k->count--;
   }
}

/* Make progress on completions: wait (until abs_timeout_ns) for kbase events,
 * read and process them. One thread reads at a time, the others wait for its
 * broadcast. Caller holds k->mtx, which is dropped while polling. Returns
 * false if nothing happened before the timeout. */
static bool
panvk_kbase_progress_locked(struct panvk_kbase_jm *k, uint64_t abs_timeout_ns)
{
   if (k->reading) {
      if (!abs_timeout_ns)
         return false;
      struct timespec ts;
      timespec_from_nsec(&ts, abs_timeout_ns);
      return u_cnd_monotonic_timedwait(&k->cnd, &k->mtx, &ts) != thrd_timedout;
   }

   k->reading = true;
   mtx_unlock(&k->mtx);

   struct pollfd pfds[2] = {
      {.fd = k->fd, .events = POLLIN},
      {.fd = k->wake_fd, .events = POLLIN},
   };
   int ret;
   do {
      struct timespec rel, *timeout = NULL;
      if (abs_timeout_ns != OS_TIMEOUT_INFINITE) {
         uint64_t now = os_time_get_nano();
         timespec_from_nsec(&rel, abs_timeout_ns > now ? abs_timeout_ns - now : 0);
         timeout = &rel;
      }
      ret = ppoll(pfds, 2, timeout, NULL);
   } while (ret < 0 && errno == EINTR);

   /* poll() said there are events and nobody else reads: read() returns
    * every pending event that fits without blocking. */
   struct panvk_kbase_event_v2 ev[32];
   ssize_t len = 0;
   bool failed = ret < 0;
   if (ret > 0 && (pfds[0].revents & POLLIN)) {
      len = read(k->fd, ev, sizeof(ev));
      failed = len <= 0;
   } else if (ret > 0 && (pfds[0].revents & (POLLERR | POLLHUP | POLLNVAL))) {
      failed = true; /* POLLERR / POLLHUP / POLLNVAL */
   }
   if (failed)
      mesa_loge("panvk: kbase event poll/read failed: %s", strerror(errno));

   mtx_lock(&k->mtx);
   for (unsigned i = 0; len > 0 && i < len / sizeof(ev[0]); i++)
      panvk_kbase_handle_event(k, &ev[i]);
   if (failed)
      panvk_kbase_mark_lost(k);
   k->reading = false;
   u_cnd_monotonic_broadcast(&k->cnd);
   return ret != 0;
}

/* P0: a WRITE_VALUE job in atom number num's slot of the barrier BO (it writes
 * num to the slot's second half). A slot is only rewritten once its atom
 * number is free again, i.e. after its previous job reported. */
static uint64_t
panvk_kbase_barrier_job(struct panvk_kbase_jm *k, uint8_t num)
{
   uint8_t *cpu = (uint8_t *)k->p0.bo->addr.host +
                  (size_t)num * PANVK_KBASE_P0_SLOT_SIZE;
   const uint64_t gpu = k->p0.bo->addr.dev +
                        (uint64_t)num * PANVK_KBASE_P0_SLOT_SIZE;

   memset(cpu, 0, PANVK_KBASE_P0_SLOT_SIZE);
   pan_section_pack(cpu, WRITE_VALUE_JOB, HEADER, cfg) {
      cfg.type = MALI_JOB_TYPE_WRITE_VALUE;
      cfg.index = 1;
   }
   pan_section_pack(cpu, WRITE_VALUE_JOB, PAYLOAD, cfg) {
      cfg.type = MALI_WRITE_VALUE_TYPE_IMMEDIATE_32;
      cfg.address = gpu + PANVK_KBASE_P0_SLOT_SIZE / 2;
      cfg.immediate_value = num;
   }
   return gpu;
}

/* P0: every submission still in the ring has reported its last atom, and
 * every export's FENCE_TRIGGER has reported. Caller holds k->mtx. */
static bool
panvk_kbase_p0_all_terminal(const struct panvk_kbase_jm *k)
{
   for (unsigned i = 0; i < k->count; i++) {
      const struct panvk_kbase_sub *sub =
         &k->subs[(k->head + i) % PANVK_KBASE_MAX_SUBS];
      if (!sub->done && !sub->terminal)
         return false;
   }
   return k->p0.aux_inflight == 0;
}

static int
panvk_kbase_p0_settle(void *data)
{
   struct panvk_kbase_jm *k = data;
   const uint64_t deadline = k->p0.progress_at_loss_ns + k->p0.deadline_ns;
   unsigned tries = 0, pending = 0;
   const char *why = NULL;

   mtx_lock(&k->mtx);
   for (unsigned i = 0; i < k->count; i++)
      pending += !k->subs[(k->head + i) % PANVK_KBASE_MAX_SUBS].terminal;

   /* 1. Every atom in flight reports. kbase bounds GPU work (hard-stop,
    * reset); only foreign fences and soft-stoppable endless chains don't. */
   while (!panvk_kbase_p0_all_terminal(k)) {
      if (os_time_get_nano() >= deadline) {
         why = "in-flight work did not report";
         break;
      }
      panvk_kbase_progress_locked(k, deadline);
   }
   const uint64_t terminal_ns = os_time_get_nano();

   /* 2. A new GPU job completes after it: kbase starts nothing on the GPU
    * while a reset is pending, so its DONE comes after any reset. */
   while (!why && !k->p0.settled) {
      if (++tries > 3) {
         why = "the barrier job failed three times";
         break;
      }
      uint8_t atom[PANVK_KBASE_ATOM_STRIDE];
      panvk_kbase_atom(atom, PANVK_KBASE_P0_ATOM,
                       panvk_kbase_barrier_job(k, PANVK_KBASE_P0_ATOM),
                       BASE_JD_REQ_CS, 0, 0, 0, 0);
      struct kbase_ioctl_job_submit submit = {
         .addr = (uint64_t)(uintptr_t)atom,
         .nr_atoms = 1,
         .stride = PANVK_KBASE_ATOM_STRIDE,
      };
      k->p0.barrier_seen = false;
      if (ioctl(k->fd, KBASE_IOCTL_JOB_SUBMIT, &submit)) {
         mesa_loge("panvk: P0 barrier submit failed: %s", strerror(errno));
         why = "the barrier job could not be submitted";
         break;
      }
      while (!k->p0.barrier_seen) {
         if (os_time_get_nano() >= deadline)
            break;
         panvk_kbase_progress_locked(k, deadline);
      }
      if (!k->p0.barrier_seen) {
         why = "the barrier job did not report";
         break;
      }
      if (k->p0.barrier_code == BASE_JD_EVENT_DONE)
         k->p0.settled = true;
      else
         mesa_loge("panvk: P0 barrier job: event 0x%x (%s), again",
                   k->p0.barrier_code, panvk_jd_event_str(k->p0.barrier_code));
   }
   const bool settled = k->p0.settled;
   mtx_unlock(&k->mtx);

   const uint64_t now = os_time_get_nano();
   if (settled) {
      uint64_t bytes = 0;
      const unsigned n =
         kbase_kmod_quarantine_release(k->dev->kmod.dev, &bytes);
      mesa_logi("panvk: P0 settled %.1f ms after the loss: %u submission(s) "
                "in flight reported after %.1f ms, barrier DONE after %.1f ms "
                "(%u tr%s); released %u freed BO(s), %" PRIu64 " KiB",
                (now - k->p0.lost_ns) / 1e6, pending,
                (terminal_ns - k->p0.lost_ns) / 1e6, (now - terminal_ns) / 1e6,
                tries, tries == 1 ? "y" : "ies", n, bytes >> 10);
   } else {
      mesa_loge("panvk: P0 NOT settled %.1f ms after the loss (%s): freed "
                "memory stays held until the device is destroyed",
                (now - k->p0.lost_ns) / 1e6, why ? why : "?");
   }
   return 0;
}

/* Latch a device loss. With P0, hold the quarantine from now on and start
 * proving the GPU done with the failed work. Caller holds k->mtx. */
static void
panvk_kbase_mark_lost(struct panvk_kbase_jm *k)
{
   k->lost = true;
   u_cnd_monotonic_broadcast(&k->cnd);
   if (!k->p0.enabled || k->p0.started)
      return;
   k->p0.started = true;
   k->p0.lost_ns = os_time_get_nano();
   k->p0.progress_at_loss_ns =
      k->last_progress_ns ? MIN2(k->last_progress_ns, k->p0.lost_ns)
                          : k->p0.lost_ns;
   kbase_kmod_quarantine_hold(k->dev->kmod.dev);
   k->p0.thread_ok =
      thrd_create(&k->p0.thread, panvk_kbase_p0_settle, k) == thrd_success;
   if (!k->p0.thread_ok)
      mesa_loge("panvk: P0 settle thread failed to start: freed memory stays "
                "held until the device is destroyed");
}

/* Caller holds k->mtx. May wait (dropping it) for completions to free
 * numbers. */
static VkResult
panvk_kbase_alloc_atoms(struct panvk_kbase_jm *k, unsigned n, uint8_t *nums,
                        enum panvk_kbase_atom_use use)
{
   assert(n > 0 && n < 256);
   while (k->nr_free < n) {
      if (k->lost || k->closing)
         return VK_ERROR_DEVICE_LOST;
      panvk_kbase_progress_locked(k, OS_TIMEOUT_INFINITE);
   }

   if (k->lost || k->closing)
      return VK_ERROR_DEVICE_LOST;

   for (unsigned i = 0; i < n; i++) {
      while (k->atom_use[k->cursor] != PANVK_KBASE_ATOM_FREE)
         k->cursor = k->cursor % 255 + 1;
      nums[i] = k->cursor;
      k->atom_use[k->cursor] = use;
      k->cursor = k->cursor % 255 + 1;
   }
   k->nr_free -= n;
   return VK_SUCCESS;
}

/* Submit n job atoms (and/or a wait for wait_fd) as one submission. Caller
 * holds k->mtx. *seqno is the new submission's, or the newest one's when
 * there is nothing to submit. */
static VkResult
panvk_kbase_submit_locked(struct panvk_kbase_jm *k,
                          const struct panvk_kbase_job *jobs, unsigned n,
                          int wait_fd, uint64_t *seqno)
{
   if (k->lost || k->closing)
      return VK_ERROR_DEVICE_LOST;

   if (!n && wait_fd < 0) {
      *seqno = k->last_seqno;
      return VK_SUCCESS;
   }

   const unsigned nr_atoms = n + (wait_fd >= 0) + (n == 0);
   uint8_t nums[PANVK_KBASE_MAX_SUB_ATOMS];
   VkResult result =
      panvk_kbase_alloc_atoms(k, nr_atoms, nums, PANVK_KBASE_ATOM_SUB);
   if (result != VK_SUCCESS)
      return result;

   /* The previous submission's last atom, while it is still in flight. */
   const uint8_t prev =
      k->count
         ? k->subs[(k->head + k->count - 1) % PANVK_KBASE_MAX_SUBS].last
         : 0;
   uint8_t atoms[PANVK_KBASE_MAX_SUB_ATOMS][PANVK_KBASE_ATOM_STRIDE];
   struct panvk_kbase_base_fence wait_fence = {.fd = wait_fd, .stream_fd = -1};
   uint8_t wait_atom = 0;
   unsigned a = 0;

   if (wait_fd >= 0) {
      k->stats.fence_wait_atoms++;
      wait_atom = nums[a];
      k->atom_info[nums[a]] = (struct panvk_kbase_atom_info){
         .core_req = BASE_JD_REQ_SOFT_FENCE_WAIT,
      };
      panvk_kbase_atom(atoms[a], nums[a], (uintptr_t)&wait_fence,
                       BASE_JD_REQ_SOFT_FENCE_WAIT |
                          BASE_JD_REQ_EVENT_ONLY_ON_FAILURE,
                       0, 0, 0, 0);
      a++;
   }

   /* ORDER on the previous submission and on the wait: a failure there does
    * not cancel this work (an errored fence was waited for like a signalled
    * one before 8.15, too). */
   for (unsigned i = 0; i < n; i++, a++) {
      uint32_t core_req = jobs[i].core_req;
      if (core_req == 0x16 && jobs[i].batch && jobs[i].batch->has_timestamp)
         core_req |= BASE_JD_REQ_PERMON;
      k->atom_info[nums[a]] = panvk_kbase_job_info(&jobs[i], k->fault_report);
      if (i + 1 < n)
         core_req |= BASE_JD_REQ_EVENT_ONLY_ON_FAILURE;
      if (i == 0)
         panvk_kbase_atom(atoms[a], nums[a], jobs[i].jc, core_req, prev,
                          BASE_JD_DEP_TYPE_ORDER,
                          k->ctrl_no_fence_wait_dep ? 0 : wait_atom,
                          BASE_JD_DEP_TYPE_ORDER);
      else if (jobs[i].overlap_second) {
         assert(i >= 2 && jobs[i].core_req == 0x16 &&
                jobs[i - 1].core_req == 0x01 &&
                jobs[i - 2].core_req == 0x16);
         panvk_kbase_atom(atoms[a], nums[a], jobs[i].jc, core_req, nums[a - 2],
                          BASE_JD_DEP_TYPE_DATA, 0, 0);
      } else if (jobs[i - 1].overlap_second) {
         assert(i >= 3 && jobs[i].core_req == 0x01);
         panvk_kbase_atom(atoms[a], nums[a], jobs[i].jc, core_req, nums[a - 1],
                          BASE_JD_DEP_TYPE_DATA, nums[a - 2],
                          BASE_JD_DEP_TYPE_DATA);
      } else
         panvk_kbase_atom(atoms[a], nums[a], jobs[i].jc, core_req, nums[a - 1],
                          BASE_JD_DEP_TYPE_DATA, 0, 0);
   }

   if (!n) {
      k->atom_info[nums[a]] = (struct panvk_kbase_atom_info){0};
      /* Only a wait: a dependency-only atom completes the submission. */
      panvk_kbase_atom(atoms[a], nums[a], 0, BASE_JD_REQ_DEP, prev,
                       BASE_JD_DEP_TYPE_ORDER, wait_atom,
                       BASE_JD_DEP_TYPE_ORDER);
      a++;
   }
   assert(a == nr_atoms);

   struct kbase_ioctl_job_submit submit = {
      .addr = (uint64_t)(uintptr_t)atoms,
      .nr_atoms = nr_atoms,
      .stride = PANVK_KBASE_ATOM_STRIDE,
   };
   if (ioctl(k->fd, KBASE_IOCTL_JOB_SUBMIT, &submit)) {
      /* Some atoms may be in the kernel: keep their numbers, give up. */
      mesa_loge("panvk: kbase JOB_SUBMIT of %u atoms failed: %s", nr_atoms,
                strerror(errno));
      panvk_kbase_mark_lost(k);
      return VK_ERROR_DEVICE_LOST;
   }

   if (!k->count)
      k->last_progress_ns = os_time_get_nano(); /* watchdog: work starts */
   assert(k->count < PANVK_KBASE_MAX_SUBS);
   struct panvk_kbase_sub *sub =
      &k->subs[(k->head + k->count) % PANVK_KBASE_MAX_SUBS];
   k->count++;
   *sub = (struct panvk_kbase_sub){
      .seqno = ++k->last_seqno,
      .last = nums[nr_atoms - 1],
      .fence_fd = -1,
      .nr_atoms = nr_atoms,
      .submit_ns = k->dev->frame_prof.enabled ? os_time_get_nano() : 0,
   };
   memcpy(sub->atoms, nums, nr_atoms);
   *seqno = sub->seqno;
   return VK_SUCCESS;
}

static VkResult
panvk_kbase_wait_seqno(struct panvk_kbase_jm *k, uint64_t seqno,
                       uint64_t abs_timeout_ns)
{
   VkResult result = VK_SUCCESS;
   const uint64_t t0 = k->dev->frame_prof.enabled ? os_time_get_nano() : 0;
   bool waited = false;

   mtx_lock(&k->mtx);
   while (k->completed_seqno < seqno) {
      waited = true;
      if (k->lost || k->closing) {
         result = VK_ERROR_DEVICE_LOST;
         break;
      }
      const bool watch = k->watchdog_ns && k->count;
      const uint64_t watch_end = k->last_progress_ns + k->watchdog_ns;
      if (!panvk_kbase_progress_locked(
             k, watch ? MIN2(abs_timeout_ns, watch_end) : abs_timeout_ns) &&
          k->completed_seqno < seqno) {
         const uint64_t now = os_time_get_nano();
         if (watch && now >= k->last_progress_ns + k->watchdog_ns &&
             !k->lost) {
            mesa_loge("panvk: no kbase event for %.1f s with %u submission(s) "
                      "in flight: device lost (PANVK_KBASE_WATCHDOG_MS)",
                      (now - k->last_progress_ns) / 1e9, k->count);
            panvk_kbase_mark_lost(k);
            vk_device_set_lost(&k->dev->vk, "kbase watchdog: no progress");
            result = VK_ERROR_DEVICE_LOST;
            break;
         }
         if (now >= abs_timeout_ns) {
            result = VK_TIMEOUT;
            break;
         }
      }
   }
   if (k->lost || k->closing)
      result = VK_ERROR_DEVICE_LOST;
   mtx_unlock(&k->mtx);
   if (t0 && waited)
      p_atomic_add(&k->dev->frame_prof.wait_ns, os_time_get_nano() - t0);
   return result;
}

#if PAN_ARCH == 9
#define PANVK_OVERLAP_SCRATCH_LIMIT (16u * 1024u * 1024u)

static bool
panvk_kbase_overlap_eligible(const struct panvk_cmd_buffer *cmdbuf,
                              const struct panvk_batch *batch)
{
   if (cmdbuf->state.uses_poly_heap || !batch->vtc_jc.first_tiler ||
       !batch->vtc_jc.first_job || !batch->frag_jc.first_job ||
       !batch->tiler.heap_desc.cpu || batch->fb.layer_count != 1 ||
       batch->tls_total_size > PANVK_OVERLAP_SCRATCH_LIMIT ||
       batch->wls_total_size > PANVK_OVERLAP_SCRATCH_LIMIT ||
       util_dynarray_num_elements(&batch->event_ops,
                                  struct panvk_cmd_event_op))
      return false;

   util_dynarray_foreach(&batch->jobs, void *, job) {
      unsigned type = (fa_rd32((const uint8_t *)*job + 16) >> 1) & 0x7f;
      if (type != MALI_JOB_TYPE_MALLOC_VERTEX && type != MALI_JOB_TYPE_FRAGMENT)
         return false;
   }

   return true;
}

static VkResult
panvk_kbase_reserve_overlap_bo(struct panvk_kbase_jm *k,
                                struct panvk_priv_bo **bo, unsigned size)
{
   if (!size || (*bo && pan_kmod_bo_size((*bo)->bo) >= size))
      return VK_SUCCESS;

   struct panvk_priv_bo *new_bo = NULL;
   VkResult result = panvk_priv_bo_create(
      k->dev, ALIGN_POT(size, 4096), PAN_KMOD_BO_FLAG_NO_MMAP,
      VK_SYSTEM_ALLOCATION_SCOPE_DEVICE, &new_bo);
   if (result != VK_SUCCESS)
      return result;

   panvk_priv_bo_unref(*bo);
   *bo = new_bo;
   return VK_SUCCESS;
}

static VkResult
panvk_kbase_prepare_overlap(struct panvk_kbase_jm *k,
                             const struct vk_queue_submit *submit)
{
   unsigned tls_size = 0, wls_size = 0;
   bool any = false;
   for (uint32_t i = 0; i < submit->command_buffer_count; i++) {
      struct panvk_cmd_buffer *cmdbuf =
         container_of(submit->command_buffers[i], struct panvk_cmd_buffer, vk);
      struct panvk_batch *previous = NULL;
      list_for_each_entry(struct panvk_batch, batch, &cmdbuf->batches, node) {
         batch->overlap_bank = 0;
         if (!panvk_kbase_overlap_eligible(cmdbuf, batch)) {
            previous = NULL;
            continue;
         }
         if (previous && previous->sync_epoch == batch->sync_epoch) {
            previous->overlap_bank = 1;
            batch->overlap_bank = 2;
            tls_size = MAX3(tls_size, previous->tls_total_size,
                            batch->tls_total_size);
            wls_size = MAX3(wls_size, previous->wls_total_size,
                            batch->wls_total_size);
            previous = NULL;
            any = true;
         } else {
            previous = batch;
         }
      }
   }
   if (!any)
      return VK_SUCCESS;

   bool grow = false;
   for (unsigned i = 0; i < 2; i++) {
      grow |= !k->overlap_banks[i].heap;
      grow |= tls_size && (!k->overlap_banks[i].tls ||
         pan_kmod_bo_size(k->overlap_banks[i].tls->bo) < tls_size);
      grow |= wls_size && (!k->overlap_banks[i].wls ||
         pan_kmod_bo_size(k->overlap_banks[i].wls->bo) < wls_size);
   }
   if (!grow)
      return VK_SUCCESS;

   mtx_lock(&k->mtx);
   uint64_t seqno = k->last_seqno;
   mtx_unlock(&k->mtx);
   VkResult result = panvk_kbase_wait_seqno(k, seqno, OS_TIMEOUT_INFINITE);
   if (result != VK_SUCCESS)
      return result;

   for (unsigned i = 0; i < 2; i++) {
      if (!k->overlap_banks[i].heap) {
         if (!i) {
            k->overlap_banks[i].heap = panvk_priv_bo_ref(k->dev->tiler_heap);
         } else {
            result = panvk_priv_bo_create(
               k->dev, pan_kmod_bo_size(k->dev->tiler_heap->bo),
               k->dev->tiler_heap->bo->flags | PAN_KMOD_BO_FLAG_NO_MMAP,
               VK_SYSTEM_ALLOCATION_SCOPE_DEVICE, &k->overlap_banks[i].heap);
            if (result != VK_SUCCESS)
               goto serialized;
         }
      }
      result = panvk_kbase_reserve_overlap_bo(
         k, &k->overlap_banks[i].tls, tls_size);
      if (result != VK_SUCCESS)
         goto serialized;
      result = panvk_kbase_reserve_overlap_bo(
         k, &k->overlap_banks[i].wls, wls_size);
      if (result != VK_SUCCESS)
         goto serialized;
   }
   return VK_SUCCESS;

serialized:
   for (uint32_t i = 0; i < submit->command_buffer_count; i++) {
      struct panvk_cmd_buffer *cmdbuf =
         container_of(submit->command_buffers[i], struct panvk_cmd_buffer, vk);
      list_for_each_entry(struct panvk_batch, batch, &cmdbuf->batches, node)
         batch->overlap_bank = 0;
   }
   return VK_SUCCESS;
}

static void
panvk_kbase_bind_overlap_bank(struct panvk_kbase_jm *k,
                               struct panvk_cmd_buffer *cmdbuf,
                               struct panvk_batch *batch)
{
   struct pan_tls_info tls = batch->tlsinfo;
   struct mali_tiler_heap_packed heap = batch->tiler.heap_templ;
   if (batch->overlap_bank) {
      unsigned i = batch->overlap_bank - 1;
      struct panvk_priv_bo *bo = k->overlap_banks[i].heap;
      if (tls.tls.size)
         tls.tls.ptr = k->overlap_banks[i].tls->addr.dev;
      if (tls.wls.size)
         tls.wls.ptr = k->overlap_banks[i].wls->addr.dev;
      pan_pack(&heap, TILER_HEAP, cfg) {
         cfg.size = pan_kmod_bo_size(bo->bo);
         cfg.base = bo->addr.dev;
         cfg.bottom = cfg.base;
         cfg.top = cfg.base + cfg.size;
         if (bo->bo->flags & PAN_KMOD_BO_FLAG_TILER_HEAP)
            cfg.top = cfg.base + MIN2(cfg.size, PAN_KMOD_TILER_HEAP_CHUNK_SIZE);
      }
   }
   if (batch->tls.cpu)
      GENX(pan_emit_tls)(&tls, batch->tls.cpu);
   if (batch->tiler.heap_desc.cpu)
      memcpy(batch->tiler.heap_desc.cpu, &heap, sizeof(heap));
   panvk_pool_flush_maps(&cmdbuf->desc_pool);
}
#endif

/* Submit what the chain holds, plus its pending wait, as one submission. */
static VkResult
panvk_kbase_flush(struct panvk_kbase_jm *k, struct panvk_kbase_chain *c,
                  bool wait, uint64_t *out_seqno)
{
   uint64_t seqno = 0;

   mtx_lock(&k->mtx);
   VkResult result =
      panvk_kbase_submit_locked(k, c->jobs, c->nr_jobs, c->wait_fd, &seqno);
   mtx_unlock(&k->mtx);

   /* kbase took its own reference to the fence during the ioctl. */
   if (c->wait_fd >= 0)
      close(c->wait_fd);
   c->wait_fd = -1;

   if (result == VK_SUCCESS) {
      for (unsigned i = 0; i < c->nr_batches; i++)
         c->batches[i]->kbase_seqno = seqno;
   }
   c->nr_jobs = 0;
   c->nr_batches = 0;

   if (result == VK_SUCCESS && wait)
      result = panvk_kbase_wait_seqno(k, seqno, OS_TIMEOUT_INFINITE);
   if (out_seqno)
      *out_seqno = seqno;
   return result;
}

/* One job chain as its own submission, waited for (the per-chain path). */
static VkResult
panvk_kbase_submit_one(struct panvk_kbase_jm *k, uint64_t jc,
                       uint32_t core_req, struct panvk_cmd_buffer *cmdbuf,
                       struct panvk_batch *batch, unsigned batch_idx,
                       uint64_t *seqno)
{
   const struct panvk_kbase_job job = {.jc = jc, .core_req = core_req,
                                       .cmdbuf = cmdbuf, .batch = batch,
                                       .batch_idx = batch_idx};

   mtx_lock(&k->mtx);
   VkResult result = panvk_kbase_submit_locked(k, &job, 1, -1, seqno);
   mtx_unlock(&k->mtx);
   if (result != VK_SUCCESS)
      return result;
   return panvk_kbase_wait_seqno(k, *seqno, OS_TIMEOUT_INFINITE);
}

/* panvk_kbase_sync_wait_func for syncs signalled by a submission. */
static VkResult
panvk_kbase_sync_wait_cb(void *data,
                         const uint64_t targets[PANVK_KBASE_SYNC_TARGET_COUNT],
                         uint64_t abs_timeout_ns)
{
   struct panvk_kbase_owner *owner = data;
   struct panvk_kbase_jm *k = panvk_kbase_owner_enter(owner);
   if (!k)
      return VK_ERROR_DEVICE_LOST;
   VkResult result = panvk_kbase_wait_seqno(k, targets[0], abs_timeout_ns);
   panvk_kbase_owner_leave(owner);
   return result;
}

/* panvk_kbase_sync_export_func: a FENCE_TRIGGER atom after the submission,
 * one per submission (later exports get a dup of the same sync_file). */
static VkResult
panvk_kbase_sync_export_active(
   void *data, const uint64_t targets[PANVK_KBASE_SYNC_TARGET_COUNT],
   int *sync_file)
{
   struct panvk_kbase_jm *k = data;
   const uint64_t seqno = targets[0];
   VkResult result = VK_SUCCESS;
   bool cpu_wait = k->cpu_fence_export;

   *sync_file = -1;
   if (k->ctrl_export_early)
      return VK_SUCCESS;

   mtx_lock(&k->mtx);
   if (k->lost || k->closing) {
      mtx_unlock(&k->mtx);
      return VK_ERROR_DEVICE_LOST;
   }
   while (!cpu_wait && k->completed_seqno < seqno) {
      struct panvk_kbase_sub *sub =
         &k->subs[(k->head + (seqno - k->subs[k->head].seqno)) %
                  PANVK_KBASE_MAX_SUBS];
      assert(k->count && sub->seqno == seqno);

      if (sub->fence_fd < 0) {
         /* P0: with the export barrier, the FENCE_TRIGGER waits for a
          * WRITE_VALUE job ORDER-dependent on the submission (a failure does
          * not cancel it): kbase only runs it once the submission is over
          * and after any GPU reset, so the sync_file never signals while the
          * GPU may still run the work (a trigger DATA-dependent on the
          * submission itself signals -EFAULT as soon as the kernel cancels
          * it, which can be during the reset). The cost: a tiny job per
          * export; the fence then reports success even if the work failed,
          * the device loss is reported through Vulkan. */
         const bool barrier = k->export_barrier;
         uint8_t nums[2];
         result = panvk_kbase_alloc_atoms(k, barrier ? 2 : 1, nums,
                                          PANVK_KBASE_ATOM_AUX);
         if (result != VK_SUCCESS)
            break;
         /* The allocation may have waited: the submission may have completed,
          * or another export may have made its fence. Look again. */
         if (k->completed_seqno >= seqno || sub->fence_fd >= 0) {
            panvk_kbase_free_atoms(k, barrier ? 2 : 1, nums);
            continue;
         }
         const uint8_t num = nums[barrier ? 1 : 0];

         struct panvk_kbase_base_fence fence = {.fd = -1, .stream_fd = -1};
         uint8_t atom[2][PANVK_KBASE_ATOM_STRIDE];
         if (barrier) {
            k->atom_info[nums[0]] =
               (struct panvk_kbase_atom_info){.core_req = BASE_JD_REQ_CS};
            panvk_kbase_atom(atom[0], nums[0],
                             panvk_kbase_barrier_job(k, nums[0]),
                             BASE_JD_REQ_CS |
                                BASE_JD_REQ_EVENT_ONLY_ON_FAILURE,
                             sub->last, BASE_JD_DEP_TYPE_ORDER, 0, 0);
         }
         k->atom_info[num] = (struct panvk_kbase_atom_info){
            .core_req = BASE_JD_REQ_SOFT_FENCE_TRIGGER};
         panvk_kbase_atom(atom[barrier ? 1 : 0], num, (uintptr_t)&fence,
                          BASE_JD_REQ_SOFT_FENCE_TRIGGER,
                          barrier ? nums[0] : sub->last,
                          BASE_JD_DEP_TYPE_DATA, 0, 0);
         struct kbase_ioctl_job_submit submit = {
            .addr = (uint64_t)(uintptr_t)atom,
            .nr_atoms = barrier ? 2 : 1,
            .stride = PANVK_KBASE_ATOM_STRIDE,
         };
         if (ioctl(k->fd, KBASE_IOCTL_JOB_SUBMIT, &submit)) {
            mesa_loge("panvk: kbase FENCE_TRIGGER submit failed: %s",
                      strerror(errno));
            panvk_kbase_mark_lost(k);
            result = VK_ERROR_DEVICE_LOST;
            break;
         }
         k->aux_pair[num] = barrier ? nums[0] : 0;
         k->p0.aux_inflight++;
         if (fence.fd < 0) {
            /* The atom completes (and frees its number) with an error. */
            mesa_loge("panvk: kbase FENCE_TRIGGER gave no fd");
            cpu_wait = true;
            break;
         }
         sub->fence_fd = fence.fd;
         k->stats.export_new++;
      } else {
         k->stats.export_dup++;
      }

      *sync_file = os_dupfd_cloexec(sub->fence_fd);
      if (*sync_file < 0)
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
      break;
   }
   if (cpu_wait)
      k->stats.export_cpu++;
   else if (*sync_file < 0 && result == VK_SUCCESS)
      k->stats.export_done++;
   mtx_unlock(&k->mtx);

   if (cpu_wait && result == VK_SUCCESS)
      result = panvk_kbase_wait_seqno(k, seqno, OS_TIMEOUT_INFINITE);
   return result;
}

static VkResult
panvk_kbase_sync_export_cb(
   void *data, const uint64_t targets[PANVK_KBASE_SYNC_TARGET_COUNT],
   int *sync_file)
{
   struct panvk_kbase_owner *owner = data;
   struct panvk_kbase_jm *k = panvk_kbase_owner_enter(owner);
   if (!k)
      return VK_ERROR_DEVICE_LOST;
   VkResult result = panvk_kbase_sync_export_active(k, targets, sync_file);
   panvk_kbase_owner_leave(owner);
   return result;
}

#if PAN_ARCH == 9
static bool
reissue_no_va_reset(void)
{
   static int cached_v = -1;
   int v = p_atomic_read_relaxed(&cached_v);
   if (v < 0) {
      v = pan_env_flag("PANVK_V9_NO_REISSUE_VA_RESET") != NULL;
      p_atomic_set(&cached_v, v);
   }
   return v;
}
#endif

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

/* PANVK_FRAME_PROF_PASSES: job counts by header type, plus the batch key. */
static void
panvk_kbase_prof_pass(struct panvk_device *dev, const struct panvk_batch *batch,
                      uint64_t vtc_ns, uint64_t frag_ns)
{
   uint32_t jobs[PANVK_FRAME_PROF_JOB_TYPES] = {0};

   util_dynarray_foreach(&batch->jobs, void *, job) {
      const uint8_t *j = *job;
      unsigned type = (fa_rd32(j + 16) >> 1) & BITFIELD_MASK(7);
      jobs[type < PANVK_FRAME_PROF_JOB_TYPES ? type : 0]++;
   }
   panvk_frame_prof_pass(dev, &batch->prof, jobs, vtc_ns, frag_ns);
}

static VkResult
panvk_queue_submit_batch(struct panvk_gpu_queue *queue,
                         struct panvk_cmd_buffer *cmdbuf,
                         struct panvk_batch *batch, uint32_t *bos,
                         unsigned nr_bos, uint32_t *in_fences,
                         unsigned nr_in_fences,
                         struct panvk_kbase_chain *chain, unsigned batch_idx)
{
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   struct panvk_physical_device *phys_dev =
      to_panvk_physical_device(dev->vk.physical);
   struct panvk_kbase_jm *k = queue->kjm;
   VkResult result;
   int ret;

   if (dev->frame_prof.enabled)
      p_atomic_inc(&dev->frame_prof.batches);

   /* Reset the batch if it's already been issued */
   if (batch->issued) {
      /* Never rewrite a batch that is still queued in the current chain, or
       * that the GPU may still be running. */
      if (chain && panvk_kbase_chain_has_batch(chain, batch)) {
         result = panvk_kbase_flush(k, chain, false, NULL);
         if (result != VK_SUCCESS)
            return result;
      }
      result = panvk_kbase_wait_seqno(k, batch->kbase_seqno,
                                      OS_TIMEOUT_INFINITE);
      if (result != VK_SUCCESS)
         return result;

      util_dynarray_foreach(&batch->jobs, void *, job) {
         memset((*job), 0, 4 * 4);
      }
#if PAN_ARCH == 9
      if (!reissue_no_va_reset()) {
         util_dynarray_foreach(&batch->vertex_jobs, void *, job) {
            uint32_t *va = (uint32_t *)((uint8_t *)(*job) +
                                        pan_section_offset(MALLOC_VERTEX_JOB,
                                                           DRAW)) + 2;
            va[0] = 1;
            va[1] = 0;
            va[2] = 0;
         }
      }
#endif

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

#if PAN_ARCH == 9
   if (k->overlap_enabled)
      panvk_kbase_bind_overlap_bank(k, cmdbuf, batch);
#endif

   /* Flush pending synchronization requests before submitting the job, to
    * make sure things are GPU-visible. */
   if (pan_kmod_flush_bo_map_syncs(dev->kmod.dev))
      return vk_device_set_lost(&dev->vk, "cache synchronization before submit failed");

   /* PATCH 0077: PANVK_FAULT_AUDIT_ALL=1 checks every batch before it goes
    * to the GPU (validation of the audit, very slow). */
   if (k->audit_all) {
      mtx_lock(&k->mtx);
      unsigned sus = 0;
      if (batch->vtc_jc.first_job)
         sus += panvk_fault_audit(k, batch, false, true,
                                  k->audit_lines < 40 ? 4 : 0);
      if (batch->frag_jc.first_job)
         sus += panvk_fault_audit(k, batch, true, true,
                                  k->audit_lines < 40 ? 4 : 0);
      k->audit_batches++;
      if (sus) {
         k->audit_suspect_batches++;
         k->audit_lines += sus;
      }
      if ((k->audit_batches & 1023) == 0)
         mesa_loge("panvk-audit: %u batches checked, %u with suspect "
                   "addresses", k->audit_batches, k->audit_suspect_batches);
      mtx_unlock(&k->mtx);
   }

   if (chain) {
      if (chain->nr_jobs + 2 > PANVK_KBASE_MAX_CHAIN ||
          chain->nr_batches + 1 > PANVK_KBASE_MAX_CHAIN) {
         result = panvk_kbase_flush(k, chain, false, NULL);
         if (result != VK_SUCCESS)
            return result;
      }
      bool overlap_second = batch->overlap_bank == 2 && chain->nr_jobs >= 2;
      if (overlap_second) {
         const struct panvk_kbase_job *last = &chain->jobs[chain->nr_jobs - 1];
         overlap_second = last->cmdbuf == cmdbuf &&
            last->batch->overlap_bank == 1 && last->core_req == 0x01 &&
            last->batch->sync_epoch == batch->sync_epoch;
      }
      if (batch->vtc_jc.first_job)
         chain->jobs[chain->nr_jobs++] = (struct panvk_kbase_job){
            .jc = batch->vtc_jc.first_job, .core_req = 0x16,
            .overlap_second = overlap_second,
            .cmdbuf = cmdbuf, .batch = batch, .batch_idx = batch_idx};
      if (batch->frag_jc.first_job)
         chain->jobs[chain->nr_jobs++] = (struct panvk_kbase_job){
            .jc = batch->frag_jc.first_job, .core_req = 0x01,
            .cmdbuf = cmdbuf, .batch = batch, .batch_idx = batch_idx};
      chain->batches[chain->nr_batches++] = batch;
      batch->issued = true;
      if (dev->frame_prof.passes)
         panvk_kbase_prof_pass(dev, batch, 0, 0);
      return VK_SUCCESS;
   }

   uint64_t prof_vtc_ns = 0, prof_frag_ns = 0;
   /* SKIP: dump sebelum submit dipastikan SIGSEGV karena ALLOC_ON_FAULT (commit_pages=0), nggak informatif */
   if (batch->vtc_jc.first_job) {
      const uint64_t pt0 = dev->frame_prof.enabled ? os_time_get_nano() : 0;
      result = panvk_kbase_submit_one(k, batch->vtc_jc.first_job, 0x16,
                                      cmdbuf, batch, batch_idx,
                                      &batch->kbase_seqno); /* PATCH eksperimen: dulu 0x16 (CS|T|V) */
      if (pt0) {
         prof_vtc_ns = os_time_get_nano() - pt0;
         p_atomic_add(&dev->frame_prof.vtc_ns, prof_vtc_ns);
      }
      ret = result == VK_SUCCESS ? 0 : -1;
      if (ret) {
         mesa_loge("panvk: v9 vtc_jc job chain failed: %d "
                   "(PANVK_V9_TRACE=1 dumps the job status)", result);
         return result;
      }
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
      const uint64_t pt0 = dev->frame_prof.enabled ? os_time_get_nano() : 0;
      result = panvk_kbase_submit_one(k, batch->frag_jc.first_job, 0x01,
                                      cmdbuf, batch, batch_idx,
                                      &batch->kbase_seqno);
      if (pt0) {
         prof_frag_ns = os_time_get_nano() - pt0;
         p_atomic_add(&dev->frame_prof.frag_ns, prof_frag_ns);
      }
      ret = result == VK_SUCCESS ? 0 : -1;
      if (ret) {
         mesa_loge("panvk: v9 frag_jc job chain failed: %d "
                   "(PANVK_V9_TRACE=1 dumps the job status)", result);
         return result;
      }
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
         if (pan_kmod_flush_bo_map_syncs(dev->kmod.dev))
            return vk_device_set_lost(&dev->vk, "descriptor cache synchronization failed");
      }

      if (PANVK_DEBUG(TRACE))
         pandecode_jc(dev->debug.decode_ctx, batch->frag_jc.first_job,
                      phys_dev->kmod.dev->props.gpu_id);

      if (PANVK_DEBUG(DUMP))
         pandecode_dump_mappings(dev->debug.decode_ctx);

   }

   if (PANVK_DEBUG(TRACE))
      pandecode_next_frame(dev->debug.decode_ctx);

   if (dev->frame_prof.passes)
      panvk_kbase_prof_pass(dev, batch, prof_vtc_ns, prof_frag_ns);

   batch->issued = true;
   return VK_SUCCESS;
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
            if (!pan_env_flag("PANVK_KBASE_EVENT_NO_SET"))
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

static VkResult
panvk_kbase_wait_external(struct panvk_kbase_jm *k, struct vk_sync *sync,
                          uint64_t value)
{
   VkResult result;
   do {
      mtx_lock(&k->mtx);
      const bool stopped = k->closing || k->lost;
      mtx_unlock(&k->mtx);
      if (stopped)
         return VK_ERROR_DEVICE_LOST;
      result = vk_sync_wait(&k->dev->vk, sync, value, VK_SYNC_WAIT_COMPLETE,
                            os_time_get_absolute_timeout(50000000));
   } while (result == VK_TIMEOUT);
   return result;
}

static VkResult
panvk_kbase_queue_submit_active(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct panvk_gpu_queue *queue = container_of(vk_queue, struct panvk_gpu_queue, vk);
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   struct panvk_kbase_jm *k = queue->kjm;
   /* One chain per submit; the per-chain path (8.9's control, trace, dump)
    * waits for every chain, so it keeps the 8.14 waits and signals too. */
   struct panvk_kbase_chain *chain = k->chain_enabled ? &k->chain : NULL;
   const bool wait_submit = k->wait_submit || !chain;
   VkResult result;
   const uint64_t prof_t0 = dev->frame_prof.enabled ? os_time_get_nano() : 0;

   panvk_mem_prof_tick(dev);

   if (pan_kmod_flush_bo_map_syncs(dev->kmod.dev))
      return vk_device_set_lost(&dev->vk, "cache synchronization before submit failed");

#if PAN_ARCH == 9
   if (k->overlap_enabled) {
      result = panvk_kbase_prepare_overlap(k, submit);
      if (result != VK_SUCCESS)
         return result;
   }
#endif

   unsigned nr_semaphores = submit->wait_count + 1;
   uint32_t semaphores[nr_semaphores];

   /* Waits. Work submitted earlier on this queue needs nothing: kbase runs
    * every submission after the previous one. Imported sync_files are merged
    * into one FENCE_WAIT atom, so the GPU waits for them. Anything else (and
    * everything under PANVK_KBASE_WAIT_SUBMIT) is waited for on the CPU
    * before submitting, as before 8.15. */
   int wait_fd = -1;
   semaphores[0] = 0;
   for (unsigned i = 0; i < submit->wait_count; i++) {
      struct vk_sync *sync = submit->waits[i].sync;
      uint64_t targets[PANVK_KBASE_SYNC_TARGET_COUNT];
      void *data = NULL;
      int fd = -1;
      enum panvk_kbase_sync_payload payload =
         wait_submit ? PANVK_KBASE_SYNC_PAYLOAD_OTHER
                     : panvk_kbase_sync_get_payload(sync, &data, targets, &fd);

      semaphores[i + 1] = 0;
      if (payload == PANVK_KBASE_SYNC_PAYLOAD_NONE) {
         k->stats.wait_none++;
         continue;
      }
      if (payload == PANVK_KBASE_SYNC_PAYLOAD_GPU && data == k->owner) {
         k->stats.wait_own++;
         continue;
      }
      if (payload == PANVK_KBASE_SYNC_PAYLOAD_FILE) {
         int ret = k->cpu_fence_wait ? -1
                                     : sync_accumulate("panvk", &wait_fd, fd);
         close(fd);
         if (!ret) {
            k->stats.wait_file++;
            continue;
         }
      }
      k->stats.wait_cpu++;

      VkResult wr = panvk_kbase_wait_external(
         k, sync, submit->waits[i].wait_value);
      if (wr != VK_SUCCESS) {
         if (wait_fd >= 0)
            close(wait_fd);
         return wr;
      }
   }
   if (chain)
      chain->wait_fd = wait_fd;
   else
      assert(wait_fd < 0);

   for (uint32_t j = 0; j < submit->command_buffer_count; ++j) {
      struct panvk_cmd_buffer *cmdbuf =
         container_of(submit->command_buffers[j], struct panvk_cmd_buffer, vk);
      unsigned batch_idx = 0;

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
          * submit what precedes it, then the batch alone, and wait for it
          * before its sets/resets are applied. */
         const bool has_event_ops = max_wait_event_syncobjs > 0;
         if (chain && has_event_ops &&
             (result = panvk_kbase_flush(k, chain, false, NULL)))
            goto err_submit;

         result = panvk_queue_submit_batch(queue, cmdbuf, batch, bos, nr_bos,
                                           in_fences, nr_in_fences, chain,
                                           batch_idx++);
         if (result != VK_SUCCESS)
            goto err_submit;

         if (chain && has_event_ops &&
             (result = panvk_kbase_flush(k, chain, true, NULL)))
            goto err_submit;

         panvk_signal_event_syncobjs(queue, batch);
      }
   }

   /* The rest of the chain (or, with no command buffers, only the waits). */
   uint64_t seqno = 0;
   if (chain) {
      result = panvk_kbase_flush(k, chain, wait_submit, &seqno);
      if (result != VK_SUCCESS)
         goto err_submit;
   } else {
      mtx_lock(&k->mtx);
      seqno = k->last_seqno;
      mtx_unlock(&k->mtx);
   }

   /* Signal fences and semaphores. Until 8.15 all work had finished here and
    * the signal was raised on the CPU (the fork's first version transferred a
    * DRM syncobj, which fails on a kbase fd: PANVK_KBASE_NO_SIGNAL=1 keeps
    * that broken behaviour as a negative control). Now a signalled sync is
    * pending on the submission's seqno: CPU waits block on completion, and
    * a SYNC_FD export becomes a FENCE_TRIGGER atom. */
   if (!k->no_signal) {
      const uint64_t targets[PANVK_KBASE_SYNC_TARGET_COUNT] = {seqno};

      for (unsigned i = 0; i < submit->signal_count; i++) {
         if (wait_submit || !seqno) {
            VkResult sr = vk_sync_signal(&dev->vk, submit->signals[i].sync,
                                         submit->signals[i].signal_value);
            if (sr != VK_SUCCESS)
               return sr;
         } else {
            panvk_kbase_sync_set_pending_owned(submit->signals[i].sync, k->owner,
                                         panvk_kbase_sync_wait_cb,
                                         panvk_kbase_sync_export_cb, targets);
         }
      }
   }

   if (prof_t0) {
      const uint64_t t1 = os_time_get_nano();
      p_atomic_add(&dev->frame_prof.submit_ns, t1 - prof_t0);
      p_atomic_inc(&dev->frame_prof.submits);
      panvk_frame_prof_tick(dev, t1);
   }
   return VK_SUCCESS;

err_submit:
   if (chain) {
      if (chain->wait_fd >= 0)
         close(chain->wait_fd);
      chain->wait_fd = -1;
      chain->nr_jobs = 0;
      chain->nr_batches = 0;
   }
   mtx_lock(&k->mtx);
   panvk_kbase_mark_lost(k);
   mtx_unlock(&k->mtx);
   return vk_device_set_lost(&dev->vk, "kbase submission failed: %d", result);
}

VkResult
panvk_per_arch(gpu_queue_submit)(struct vk_queue *vk_queue,
                                  struct vk_queue_submit *submit)
{
   struct panvk_gpu_queue *queue =
      container_of(vk_queue, struct panvk_gpu_queue, vk);
   struct panvk_kbase_owner *owner = queue->kjm->owner;
   if (!panvk_kbase_owner_enter(owner))
      return VK_ERROR_DEVICE_LOST;
   VkResult result = panvk_kbase_queue_submit_active(vk_queue, submit);
   panvk_kbase_owner_leave(owner);
   return result;
}

static void
panvk_kbase_jm_close(struct panvk_kbase_jm *k)
{
   mtx_lock(&k->mtx);
   k->closing = true;
   const uint64_t wake = 1;
   ssize_t ret;
   do {
      ret = write(k->wake_fd, &wake, sizeof(wake));
   } while (ret < 0 && errno == EINTR);
   u_cnd_monotonic_broadcast(&k->cnd);
   mtx_unlock(&k->mtx);
   panvk_kbase_owner_close(k->owner);
}

static void
panvk_kbase_log_stats(struct panvk_kbase_jm *k)
{
   if (!k->trace)
      return;
   mtx_lock(&k->mtx);
   mesa_logi("panvk: kbase submissions %" PRIu64 "; waits: none %" PRIu64
             " own %" PRIu64 " sync_file %" PRIu64 " cpu %" PRIu64
             "; FENCE_WAIT atoms %" PRIu64 "; exports: new %" PRIu64
             " dup %" PRIu64 " done %" PRIu64 " cpu %" PRIu64,
             k->last_seqno, k->stats.wait_none, k->stats.wait_own,
             k->stats.wait_file, k->stats.wait_cpu, k->stats.fence_wait_atoms,
             k->stats.export_new, k->stats.export_dup, k->stats.export_done,
             k->stats.export_cpu);
   mtx_unlock(&k->mtx);
}

static VkResult
panvk_kbase_jm_create(struct panvk_device *device, struct panvk_kbase_jm **out)
{
   struct panvk_kbase_jm *k =
      vk_zalloc(&device->vk.alloc, sizeof(*k), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!k)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   k->fd = device->drm_fd;
   k->chain.wait_fd = -1;
   /* Atom number 0 means "no dependency": never hand it out. 255 is the P0
    * settle barrier's. */
   k->atom_use[0] = PANVK_KBASE_ATOM_SUB;
   k->atom_use[PANVK_KBASE_P0_ATOM] = PANVK_KBASE_ATOM_SUB;
   k->nr_free = 254;
   k->cursor = 1;
   k->wait_submit = pan_env_flag("PANVK_KBASE_WAIT_SUBMIT") != NULL;
   k->cpu_fence_wait = pan_env_flag("PANVK_KBASE_CPU_FENCE_WAIT") != NULL;
   k->cpu_fence_export = pan_env_flag("PANVK_KBASE_CPU_FENCE_EXPORT") != NULL;
   k->trace = pan_env_flag("PANVK_KBASE_FENCE_TRACE") != NULL;
   k->ctrl_no_fence_wait_dep =
      pan_env_flag("PANVK_KBASE_CTRL_NO_FENCE_WAIT_DEP") != NULL;
   k->ctrl_export_early = pan_env_flag("PANVK_KBASE_CTRL_EXPORT_EARLY") != NULL;
   k->dev = device;
   k->chain_enabled = panvk_kbase_chain_enabled();
#if PAN_ARCH == 9
   k->overlap_enabled = k->chain_enabled &&
      debug_get_bool_option("PANVK_JM_OVERLAP", false);
#endif
   k->no_signal = pan_env_flag("PANVK_KBASE_NO_SIGNAL") != NULL;
   k->t0_ns = os_time_get_nano();
   k->fault_report = pan_env_flag("PANVK_FAULT_REPORT") != NULL;
   k->audit_all = pan_env_flag("PANVK_FAULT_AUDIT_ALL") != NULL;
   /* P0 (see PANVK_KBASE_P0_ATOM): on by default; =0 is the control. */
   k->p0.enabled = debug_get_bool_option("PANVK_KBASE_P0", true);
   k->p0.deadline_ns =
      debug_get_num_option("PANVK_KBASE_P0_DEADLINE_MS", 12000) * 1000000ull;
   k->watchdog_ns =
      debug_get_num_option("PANVK_KBASE_WATCHDOG_MS", 30000) * 1000000ull;
   if (k->p0.enabled || debug_get_bool_option("PANVK_KBASE_EXPORT_BARRIER",
                                              false)) {
      if (panvk_priv_bo_create(device, 256 * PANVK_KBASE_P0_SLOT_SIZE, 0,
                               VK_SYSTEM_ALLOCATION_SCOPE_DEVICE,
                               &k->p0.bo) != VK_SUCCESS) {
         mesa_loge("panvk: P0 barrier BO allocation failed: P0 off");
         k->p0.bo = NULL;
         k->p0.enabled = false;
      }
   }
   k->export_barrier =
      k->p0.bo && debug_get_bool_option("PANVK_KBASE_EXPORT_BARRIER",
                                        k->p0.enabled);

   if (mtx_init(&k->mtx, mtx_plain) != thrd_success)
      goto err_free;
   if (u_cnd_monotonic_init(&k->cnd) != thrd_success)
      goto err_mtx;

   k->wake_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
   if (k->wake_fd < 0)
      goto err_cnd;
   k->owner = panvk_kbase_owner_create(k);
   if (!k->owner) {
      close(k->wake_fd);
      goto err_cnd;
   }

   *out = k;
   return VK_SUCCESS;

err_cnd:
   u_cnd_monotonic_destroy(&k->cnd);
err_mtx:
   mtx_destroy(&k->mtx);
err_free:
   panvk_priv_bo_unref(k->p0.bo);
   vk_free(&device->vk.alloc, k);
   return panvk_error(device, VK_ERROR_INITIALIZATION_FAILED);
}

static void
panvk_kbase_jm_destroy(struct panvk_device *device, struct panvk_kbase_jm *k)
{
   if (!k)
      return;

   /* P0: let the settle finish (it ends by its deadline) before anything is
    * torn down; what is still held after that is freed at device destroy. */
   if (k->p0.started && k->p0.thread_ok)
      thrd_join(k->p0.thread, NULL);

   panvk_kbase_jm_close(k);
   uint64_t wake;
   while (read(k->wake_fd, &wake, sizeof(wake)) < 0 && errno == EINTR)
      ;
   mtx_lock(&k->mtx);
   assert(!k->reading);
   const uint64_t timeout = os_time_get_absolute_timeout(
      10ull * 1000 * 1000 * 1000);
   while (!k->lost && k->completed_seqno < k->last_seqno) {
      if (!panvk_kbase_progress_locked(k, timeout))
         break;
   }
   if (k->lost ? !k->p0.settled : k->completed_seqno < k->last_seqno)
      mesa_loge("panvk: kbase GPU retirement unproven at destroy%s",
                k->lost && !k->p0.enabled ? " (PANVK_KBASE_P0=0)" : "");
   mtx_unlock(&k->mtx);

   panvk_kbase_log_stats(k);

   for (unsigned i = 0; i < k->count; i++) {
      struct panvk_kbase_sub *sub =
         &k->subs[(k->head + i) % PANVK_KBASE_MAX_SUBS];
      if (sub->fence_fd >= 0)
         close(sub->fence_fd);
   }
   if (k->chain.wait_fd >= 0)
      close(k->chain.wait_fd);
   panvk_priv_bo_unref(k->p0.bo);
   for (unsigned i = 0; i < 2; i++) {
      panvk_priv_bo_unref(k->overlap_banks[i].heap);
      panvk_priv_bo_unref(k->overlap_banks[i].tls);
      panvk_priv_bo_unref(k->overlap_banks[i].wls);
   }
   close(k->wake_fd);
   panvk_kbase_owner_unref(k->owner);
   u_cnd_monotonic_destroy(&k->cnd);
   mtx_destroy(&k->mtx);
   vk_free(&device->vk.alloc, k);
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

   /* JM has no queue priorities: LOW is accepted and runs at MEDIUM
    * (check_global_priority). */
   assert(priority <= VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR);

   struct panvk_gpu_queue *queue = vk_zalloc(&device->vk.alloc, sizeof(*queue), 8,
                                         VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!queue)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      vk_queue_init(&queue->vk, &device->vk, create_info, queue_idx);
   if (result != VK_SUCCESS)
      goto err_free_queue;

   queue->sync = 0;

   /* Atom numbers belong to the kbase context (one per VkDevice), and v9
    * exposes a single queue: the state lives with it. */
   result = panvk_kbase_jm_create(device, &queue->kjm);
   if (result != VK_SUCCESS)
      goto err_finish_queue;

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

   panvk_kbase_jm_close(queue->kjm);
   vk_queue_finish(&queue->vk);
   panvk_kbase_jm_destroy(dev, queue->kjm);
   /* PATCH: no drm_syncobj to destroy */
   vk_free(&dev->vk.alloc, queue);
}

VkResult
panvk_per_arch(gpu_queue_check_status)(struct vk_queue *vk_queue)
{
   return VK_SUCCESS;
}

static VkResult
panvk_kbase_queue_wait_idle_active(VkQueue _queue)
{
   VK_FROM_HANDLE(panvk_gpu_queue, queue, _queue);
   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   struct panvk_kbase_jm *k = queue->kjm;

   /* we need to use vk_common_QueueWaitIdle if we ever go threaded */
   assert(queue->vk.submit.mode != VK_QUEUE_SUBMIT_MODE_THREADED);

   if (vk_device_is_lost(&dev->vk)) {
      /* Check printf buffer one more time before exiting */
      u_printf_with_ctx(stdout, &dev->printf.ctx);
      return VK_ERROR_DEVICE_LOST;
   }

   /* Submits no longer wait for their work (8.15): wait for the newest. */
   uint64_t seqno;
   mtx_lock(&k->mtx);
   seqno = k->last_seqno;
   mtx_unlock(&k->mtx);
   VkResult result = panvk_kbase_wait_seqno(k, seqno, OS_TIMEOUT_INFINITE);
   if (result != VK_SUCCESS)
      return vk_device_set_lost(&dev->vk, "kbase queue wait failed: %d",
                                result);
   panvk_kbase_log_stats(k);

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(QueueWaitIdle)(VkQueue _queue)
{
   VK_FROM_HANDLE(panvk_gpu_queue, queue, _queue);
   struct panvk_kbase_owner *owner = queue->kjm->owner;
   if (!panvk_kbase_owner_enter(owner))
      return VK_ERROR_DEVICE_LOST;
   VkResult result = panvk_kbase_queue_wait_idle_active(_queue);
   panvk_kbase_owner_leave(owner);
   return result;
}
