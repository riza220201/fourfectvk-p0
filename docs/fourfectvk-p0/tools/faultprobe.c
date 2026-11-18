/* faultprobe.c -- raw kbase JM fault injection on the RS4 (mesa-mobile JOURNAL (i)/(j), session 2).
 *
 * Asks the phone's kbase (r54p1, BM_2 atom stride 64) directly, without Mesa, what a driver gets back
 * when a submission fails or never ends. Built on ~/panvk-g99/fenceprobe/fence_probe.c (10-06).
 *
 *   f1 [tf|pf]          MMU fault among in-flight atoms (tf: write to a freed BO's VA = translation fault,
 *                       pf: write to a GPU-read-only BO = permission fault). Context A: 4 good, the bad one,
 *                       one DATA-dependent and one ORDER-dependent on it, 4 good; context B: 8 good at the
 *                       same time. Every atom must report exactly once; then A is reused (same atom numbers).
 *   f2 <secs> <end>     A job chain that never ends but obeys soft-stop: two WRITE_VALUE jobs that clear each
 *                       other's status and jump to each other (see loop_chain). Watch <secs> for an event, then
 *                       end it: "gpucut" (a second atom in the same context zeroes the ring's next pointer from
 *                       the GPU side; the loop should end DONE), "close" (time the context teardown with the
 *                       loop running), "none" (leave it to process exit).
 *   f3 <secs>           FENCE_WAIT on foreign fences. (a) control: a fence held by an EVENT_WAIT in context B
 *                       (soft_job_timeout applies to EVENT_WAIT); (b) a fence behind B's never-ending chain,
 *                       watched <secs>; (c) context C waits on such a fence and is torn down while waiting;
 *                       (d) the producer B is torn down: A's wait must then resolve.
 *   f4 <max secs> [fence]  An infinite-loop compute shader (stores a counter each pass): soft-stop cannot take it off
 *                       the GPU. Wait for its event (hard-stop or reset), free the store target at once, reuse the
 *                       context. Run under faultrun.sh with FR_JS_TIMEOUTS to force the reset path ("race").
 *   f5 <max secs>       The P0 barrier premise: hang -> ORDER -> WRITE_VALUE barrier -> DATA -> FENCE_TRIGGER.
 *   wv <type> <loop>    isolation: one WRITE_VALUE of <type>, optionally jumping to itself.
 *
 * Every event is printed with CLOCK_MONOTONIC seconds and written to the ftrace marker, so kprobes recorded
 * with trace_clock=mono line up with it. */
#include <errno.h>
#include <math.h>
#include <fcntl.h>
#include <poll.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <linux/sync_file.h>

#define KBASE_IOCTL_TYPE 0x80
struct kbase_ioctl_version_check { uint16_t major, minor; };
#define KBASE_IOCTL_VERSION_CHECK _IOWR(KBASE_IOCTL_TYPE, 0, struct kbase_ioctl_version_check)
struct kbase_ioctl_set_flags { uint32_t create_flags; };
#define KBASE_IOCTL_SET_FLAGS _IOW(KBASE_IOCTL_TYPE, 1, struct kbase_ioctl_set_flags)
struct kbase_ioctl_job_submit { uint64_t addr; uint32_t nr_atoms, stride; };
#define KBASE_IOCTL_JOB_SUBMIT _IOW(KBASE_IOCTL_TYPE, 2, struct kbase_ioctl_job_submit)
union kbase_ioctl_mem_alloc {
   struct { uint64_t va_pages, commit_pages, extension, flags; } in;
   struct { uint64_t flags, gpu_va; } out;
};
#define KBASE_IOCTL_MEM_ALLOC _IOWR(KBASE_IOCTL_TYPE, 5, union kbase_ioctl_mem_alloc)
struct kbase_ioctl_mem_free { uint64_t gpu_addr; };
#define KBASE_IOCTL_MEM_FREE _IOW(KBASE_IOCTL_TYPE, 7, struct kbase_ioctl_mem_free)
struct kbase_ioctl_mem_exec_init { uint64_t va_pages; };
#define KBASE_IOCTL_MEM_EXEC_INIT _IOW(KBASE_IOCTL_TYPE, 38, struct kbase_ioctl_mem_exec_init)
#define BASE_MEM_PROT_GPU_EX (1ull << 4)

#define BASE_MEM_PROT_CPU_RD (1ull << 0)
#define BASE_MEM_PROT_CPU_WR (1ull << 1)
#define BASE_MEM_PROT_GPU_RD (1ull << 2)
#define BASE_MEM_PROT_GPU_WR (1ull << 3)
#define BASE_MEM_COHERENT_LOCAL (1ull << 11)
#define BASE_MEM_MAP_TRACKING_HANDLE (3ull << 12)
#define RW (BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_CPU_WR | BASE_MEM_PROT_GPU_RD | BASE_MEM_PROT_GPU_WR)

#define BASE_JD_REQ_CS (1u << 1)
#define BASE_JD_REQ_SOFT_JOB (1u << 9)
#define BASE_JD_REQ_SOFT_FENCE_TRIGGER (BASE_JD_REQ_SOFT_JOB | 0x2)
#define BASE_JD_REQ_SOFT_FENCE_WAIT (BASE_JD_REQ_SOFT_JOB | 0x3)
#define BASE_JD_REQ_SOFT_EVENT_WAIT (BASE_JD_REQ_SOFT_JOB | 0x5)
#define DEP_DATA 1
#define DEP_ORDER 2

struct base_dependency { uint8_t atom_id, dependency_type; };
struct atom64 {                         /* base_jd_atom_v2 + BM_2 frame_nr */
   uint64_t jc;
   uint64_t udata[2];
   uint64_t extres_list;
   uint16_t nr_extres;
   uint8_t jit_id[2];
   struct base_dependency pre_dep[2];
   uint8_t atom_number, prio, device_nr, jobslot;
   uint32_t core_req;
   uint8_t padding[8];
   uint32_t frame_nr;
   uint32_t pad2;
};
_Static_assert(sizeof(struct atom64) == 64, "stride 64");
_Static_assert(offsetof(struct atom64, pre_dep) == 36, "pre_dep@36");
_Static_assert(offsetof(struct atom64, atom_number) == 40, "atom_number@40");
_Static_assert(offsetof(struct atom64, core_req) == 44, "core_req@44");

struct base_fence { struct { int fd, stream_fd; } basep; };
struct jd_event { uint32_t event_code; uint8_t atom_number; uint8_t pad[3]; uint64_t udata[2]; };

static int marker_fd = -1;
static double t0_s;

static double now_s(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* printf to stdout, prefixed with the time since start, and the same text to the ftrace marker. */
static void say(const char *fmt, ...)
{
   char buf[512];
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(buf, sizeof(buf), fmt, ap);
   va_end(ap);
   double t = now_s();
   printf("[%10.6f +%8.3f] %s\n", t, t - t0_s, buf);
   fflush(stdout);
   if (marker_fd >= 0)
      dprintf(marker_fd, "faultprobe: %s\n", buf);
}

static const char *code_name(uint32_t c)
{
   switch (c) {
   case 0x01: return "DONE";
   case 0x03: return "STOPPED";
   case 0x04: return "TERMINATED";
   case 0x40: return "JOB_CONFIG_FAULT";
   case 0x42: return "JOB_READ_FAULT";
   case 0x43: return "JOB_WRITE_FAULT";
   case 0x48: return "JOB_BUS_FAULT";
   case 0x58: return "DATA_INVALID_FAULT";
   case 0x5A: return "STATE_FAULT";
   case 0x7F: return "UNKNOWN";
   case 0x80: return "DELAYED_BUS_FAULT";
   case 0xC1: case 0xC2: case 0xC3: case 0xC4: return "TRANSLATION_FAULT";
   case 0xC8: return "PERMISSION_FAULT";
   case 0xD8: return "ACCESS_FLAG";
   case 0x4000: return "MEM_GROWTH_FAILED";
   case 0x4002: return "JOB_CANCELLED";
   case 0x4003: return "JOB_INVALID";
   default: return "?";
   }
}

struct ctx {
   char name;
   int fd;
   size_t tgt_sz, desc_sz;
   volatile uint32_t *tgt;  /* GPU-written 32-bit values, 4 pages */
   uint8_t *desc;           /* job descriptors, 128 B per slot, 8 pages = 256 slots */
   uint8_t *evt;            /* soft events */
   volatile uint64_t *ts;   /* GPU-written timestamps */
   void *tracking;          /* kbase keeps the context alive while any mapping exists */
   void *extra[4];
   size_t extra_sz[4];
   unsigned n_extra;
   uint32_t ev_code[256];
   int ev_count[256];
   double ev_t[256];
};

static void *alloc_map(int fd, unsigned pages, uint64_t flags, uint64_t *va_out)
{
   union kbase_ioctl_mem_alloc a = {0};
   a.in.va_pages = pages;
   a.in.commit_pages = pages;
   a.in.flags = flags;
   if (ioctl(fd, KBASE_IOCTL_MEM_ALLOC, &a)) {
      printf("MEM_ALLOC(%u pages, 0x%llx) failed: %s\n", pages, (unsigned long long)flags, strerror(errno));
      exit(1);
   }
   void *p = mmap(NULL, 4096u * pages, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)a.out.gpu_va);
   if (p == MAP_FAILED) {
      /* a GPU-only region still needs its cookie mapped once to get a VA; retry read-only */
      p = mmap(NULL, 4096u * pages, PROT_READ, MAP_SHARED, fd, (off_t)a.out.gpu_va);
      if (p == MAP_FAILED) {
         printf("mmap failed: %s\n", strerror(errno));
         exit(1);
      }
   }
   if (va_out)
      *va_out = (uint64_t)(uintptr_t)p;
   return p;
}

static void ctx_open(struct ctx *c, char name)
{
   memset(c, 0, sizeof(*c));
   c->name = name;
   c->fd = open("/dev/mali0", O_RDWR | O_CLOEXEC);
   if (c->fd < 0) { printf("open: %s\n", strerror(errno)); exit(1); }
   struct kbase_ioctl_version_check ver = {0, 0};
   ioctl(c->fd, KBASE_IOCTL_VERSION_CHECK, &ver);
   struct kbase_ioctl_set_flags sf = {0};
   ioctl(c->fd, KBASE_IOCTL_SET_FLAGS, &sf);
   c->tracking = mmap(NULL, 4096, PROT_NONE, MAP_SHARED, c->fd, BASE_MEM_MAP_TRACKING_HANDLE);
   if (c->tracking == MAP_FAILED)
      printf("tracking mmap: %s\n", strerror(errno));
   struct kbase_ioctl_mem_exec_init ei = { 1024 };   /* 4 MiB EXEC_VA zone, before any allocation */
   if (ioctl(c->fd, KBASE_IOCTL_MEM_EXEC_INIT, &ei))
      printf("MEM_EXEC_INIT: %s\n", strerror(errno));
   c->tgt_sz = 4 * 4096;
   c->desc_sz = 8 * 4096;
   c->tgt = alloc_map(c->fd, 4, BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_CPU_WR | BASE_MEM_PROT_GPU_WR |
                                   BASE_MEM_COHERENT_LOCAL, NULL);
   c->desc = alloc_map(c->fd, 8, RW, NULL);
   c->evt = alloc_map(c->fd, 1, RW, NULL);
   c->ts = alloc_map(c->fd, 1, BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_CPU_WR | BASE_MEM_PROT_GPU_WR |
                                  BASE_MEM_COHERENT_LOCAL, NULL);
   memset(c->desc, 0, c->desc_sz);
   memset(c->evt, 0, 4096);
   for (unsigned i = 0; i < c->tgt_sz / 4; i++)
      c->tgt[i] = 0xDEADBEEF;
   for (unsigned i = 0; i < 512; i++)
      c->ts[i] = 0;
   say("ctx %c fd=%d kbase UK %u.%u tgt=%p desc=%p", name, c->fd, ver.major, ver.minor, (void *)c->tgt,
       (void *)c->desc);
}

static double close_ctx(struct ctx *c)
{
   double t = now_s();
   munmap((void *)c->tgt, c->tgt_sz);
   munmap(c->desc, c->desc_sz);
   munmap(c->evt, 4096);
   munmap((void *)c->ts, 4096);
   for (unsigned i = 0; i < c->n_extra; i++)
      munmap(c->extra[i], c->extra_sz[i]);
   munmap(c->tracking, 4096);
   close(c->fd);   /* last reference: kbase destroys the context on the way back to userspace */
   return now_s() - t;
}

static uint8_t *slot(struct ctx *c, unsigned s) { return c->desc + 128 * s; }
static uint64_t slot_va(struct ctx *c, unsigned s) { return (uint64_t)(uintptr_t)slot(c, s); }

static void put64(uint32_t *d, unsigned w, uint64_t v) { d[w] = (uint32_t)v; d[w + 1] = (uint32_t)(v >> 32); }

/* Job header (v9): w4 = type << 1 | barrier << 8 | index << 16; w6..7 = next. */
static void header(uint32_t *d, unsigned type, unsigned index, uint64_t next)
{
   memset(d, 0, 128);
   d[4] = (type << 1) | (index << 16);
   put64(d, 6, next);
}

/* WRITE_VALUE: payload at word 8: address (64), type (w10), immediate (w12..13). */
static uint64_t wv_job(struct ctx *c, unsigned s, uint64_t addr, unsigned type, uint64_t value, uint64_t next)
{
   uint32_t *d = (uint32_t *)slot(c, s);
   header(d, 2, 1, next);
   put64(d, 8, addr);
   d[10] = type;     /* 2 = system timestamp (64-bit), 6 = immediate 32 */
   put64(d, 12, value);
   return slot_va(c, s);
}

static uint64_t write32_job(struct ctx *c, unsigned s, unsigned tgt_idx, uint32_t value)
{
   return wv_job(c, s, (uint64_t)(uintptr_t)&c->tgt[tgt_idx], 6, value, 0);
}

/* The never-ending chain in slots s, s+1. The job manager refuses a job whose header already carries an
 * exception status (a job met twice: DATA_INVALID_FAULT, status 0x00010258), so the two jobs clear each
 * other: J0 = WRITE_VALUE(imm64 0 -> J1's status words) -> J1 = WRITE_VALUE(imm64 0 -> J0's) -> J0 ...
 * Both need Suppress Prefetch (w4 bit 11): with prefetch the JM reads J0's header while J1 runs, before
 * J1's clear, and faults the same way. The descriptors then live in the GPU's L2: the CPU sees neither the
 * status words change nor can it cut the chain by writing memory; liveness comes from the kernel trace.
 * FP_FLAGS overrides the extra w4 bits (default 0x800). */
static uint64_t loop_chain(struct ctx *c, unsigned s)
{
   uint64_t j0 = slot_va(c, s), j1 = slot_va(c, s + 1);
   unsigned fl = getenv("FP_FLAGS") ? strtoul(getenv("FP_FLAGS"), NULL, 0) : 0x800;
   wv_job(c, s, j1, 7, 0, j1);
   wv_job(c, s + 1, j0, 7, 0, j0);
   ((uint32_t *)slot(c, s))[4] = (2u << 1) | fl | (1u << 16);
   ((uint32_t *)slot(c, s + 1))[4] = (2u << 1) | fl | (2u << 16);
   __sync_synchronize();
   return j0;
}

/* A GPU-side cut: a WRITE_VALUE that zeroes J1's next pointer (word 6) through the GPU's own caches. */
static uint64_t cut_job(struct ctx *c, unsigned s_cut, unsigned s_loop)
{
   return wv_job(c, s_cut, slot_va(c, s_loop + 1) + 24, 7, 0, 0);
}

static void atom(struct atom64 *a, uint8_t num, uint64_t jc, uint32_t core_req,
                 uint8_t dep0, uint8_t type0, uint8_t dep1, uint8_t type1)
{
   memset(a, 0, sizeof(*a));
   a->jc = jc;
   a->atom_number = num;
   a->core_req = core_req;
   if (dep0) { a->pre_dep[0].atom_id = dep0; a->pre_dep[0].dependency_type = type0; }
   if (dep1) { a->pre_dep[1].atom_id = dep1; a->pre_dep[1].dependency_type = type1; }
}

static int submit(struct ctx *c, struct atom64 *atoms, unsigned n)
{
   struct kbase_ioctl_job_submit s = { (uint64_t)(uintptr_t)atoms, n, sizeof(struct atom64) };
   if (ioctl(c->fd, KBASE_IOCTL_JOB_SUBMIT, &s)) {
      say("ctx %c JOB_SUBMIT of %u atoms failed: %s", c->name, n, strerror(errno));
      return -1;
   }
   say("ctx %c submitted %u atoms (first %u)", c->name, n, atoms[0].atom_number);
   return 0;
}

static void clear_events(struct ctx *c)
{
   memset(c->ev_code, 0, sizeof(c->ev_code));
   memset(c->ev_count, 0, sizeof(c->ev_count));
   memset(c->ev_t, 0, sizeof(c->ev_t));
}

/* Read every event available within timeout_ms (0 = only what is pending). Returns how many. */
static int read_events(struct ctx *c, int timeout_ms)
{
   int n = 0;
   for (;;) {
      struct pollfd p = { c->fd, POLLIN, 0 };
      if (poll(&p, 1, n ? 0 : timeout_ms) <= 0)
         return n;
      struct jd_event e;
      if (read(c->fd, &e, sizeof(e)) != sizeof(e))
         return n;
      double t = now_s();
      c->ev_count[e.atom_number]++;
      if (c->ev_count[e.atom_number] == 1)
         c->ev_t[e.atom_number] = t;
      c->ev_code[e.atom_number] = e.event_code;
      say("ctx %c EVENT atom %u code 0x%x %s (#%d for this atom)", c->name, e.atom_number, e.event_code,
          code_name(e.event_code), c->ev_count[e.atom_number]);
      n++;
   }
}

/* Wait until every atom in nums[] has reported, or the deadline. Returns how many are still missing. */
static int wait_atoms(struct ctx *c, const uint8_t *nums, unsigned n, double timeout_s)
{
   double end = now_s() + timeout_s;
   for (;;) {
      unsigned missing = 0;
      for (unsigned i = 0; i < n; i++)
         missing += c->ev_count[nums[i]] == 0;
      if (!missing)
         return 0;
      int left = (int)((end - now_s()) * 1000);
      if (left <= 0)
         return missing;
      read_events(c, left > 200 ? 200 : left);
   }
}

static int fails;
static void check(int ok, const char *what)
{
   say("  %s %s", ok ? "OK  " : "FAIL", what);
   if (!ok)
      fails++;
}

static int fence_status(int fd)
{
   struct sync_fence_info fi[2];
   struct sync_file_info info = {0};
   info.num_fences = 2;
   info.sync_fence_info = (uint64_t)(uintptr_t)fi;
   if (ioctl(fd, SYNC_IOC_FILE_INFO, &info))
      return -1000;
   return info.status;
}

/* ------------------------------------------------------------------------------------------------ f1 */
static void test_f1(const char *kind)
{
   int pf = kind && !strcmp(kind, "pf");
   say("F1 MMU fault (%s) among in-flight atoms", pf ? "permission" : "translation");
   struct ctx A, B;
   ctx_open(&A, 'A');
   ctx_open(&B, 'B');

   uint64_t bad;
   if (pf) {
      A.extra[A.n_extra] = alloc_map(A.fd, 1, BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_GPU_RD, &bad); /* GPU reads only */
      A.extra_sz[A.n_extra++] = 4096;
      say("bad VA 0x%llx = a GPU-read-only BO", (unsigned long long)bad);
   } else {
      void *p = alloc_map(A.fd, 1, RW, &bad);
      struct kbase_ioctl_mem_free mf = { bad };
      int r = ioctl(A.fd, KBASE_IOCTL_MEM_FREE, &mf);
      int e = errno;
      munmap(p, 4096);
      say("bad VA 0x%llx = a freed BO (MEM_FREE rc %d %s)", (unsigned long long)bad, r, r ? strerror(e) : "");
   }

   struct atom64 at[16];
   uint8_t nums[11];
   for (unsigned i = 0; i < 11; i++)
      nums[i] = i + 1;
   for (unsigned i = 1; i <= 4; i++)
      atom(&at[i - 1], i, write32_job(&A, i, i, 0x1000 + i), BASE_JD_REQ_CS, 0, 0, 0, 0);
   atom(&at[4], 5, wv_job(&A, 5, bad, 6, 0x5555, 0), BASE_JD_REQ_CS, 0, 0, 0, 0);
   atom(&at[5], 6, write32_job(&A, 6, 6, 0x1006), BASE_JD_REQ_CS, 5, DEP_DATA, 0, 0);
   atom(&at[6], 7, write32_job(&A, 7, 7, 0x1007), BASE_JD_REQ_CS, 5, DEP_ORDER, 0, 0);
   for (unsigned i = 8; i <= 11; i++)
      atom(&at[i - 1], i, write32_job(&A, i, i, 0x1000 + i), BASE_JD_REQ_CS, 0, 0, 0, 0);

   struct atom64 bt[8];
   uint8_t bnums[8];
   for (unsigned i = 1; i <= 8; i++) {
      bnums[i - 1] = i;
      atom(&bt[i - 1], i, write32_job(&B, i, i, 0x2000 + i), BASE_JD_REQ_CS, 0, 0, 0, 0);
   }

   double ts = now_s();
   check(submit(&A, at, 11) == 0, "A: 11 atoms submitted");
   check(submit(&B, bt, 8) == 0, "B: 8 atoms submitted");
   int ma = wait_atoms(&A, nums, 11, 15.0);
   int mb = wait_atoms(&B, bnums, 8, 5.0);
   say("A: %d atoms without an event after %.3f s; B: %d", ma, now_s() - ts, mb);
   usleep(1000000);
   read_events(&A, 0);
   read_events(&B, 0);

   int once = 1;
   for (unsigned i = 1; i <= 11; i++) {
      say("  A atom %2u: %d event(s), code 0x%x %s at +%.3f ms, tgt=0x%08x", i, A.ev_count[i], A.ev_code[i],
          code_name(A.ev_code[i]), A.ev_count[i] ? (A.ev_t[i] - ts) * 1e3 : -1.0, i == 5 ? 0 : A.tgt[i]);
      once &= A.ev_count[i] == 1;
   }
   check(once, "A: every atom reported exactly once");
   int bok = 1;
   for (unsigned i = 1; i <= 8; i++)
      bok &= B.ev_count[i] == 1 && B.ev_code[i] == 1 && B.tgt[i] == 0x2000 + i;
   check(bok, "B (other context): all 8 DONE once, values written");
   int wr_ok = 1;
   for (unsigned i = 1; i <= 11; i++) {
      if (i == 5)
         continue;
      int done = A.ev_code[i] == 1, written = A.tgt[i] == 0x1000 + i;
      wr_ok &= done == written;
   }
   check(wr_ok, "A: a value is written exactly when its atom reported DONE");

   say("F1 phase 2: reuse context A (same atom numbers 1..4, then 20..23)");
   clear_events(&A);
   for (unsigned i = 1; i <= 4; i++) {
      A.tgt[100 + i] = 0xDEADBEEF;
      atom(&at[i - 1], i, write32_job(&A, 20 + i, 100 + i, 0x3000 + i), BASE_JD_REQ_CS, 0, 0, 0, 0);
      atom(&at[4 + i - 1], 20 + i, write32_job(&A, 30 + i, 110 + i, 0x3100 + i), BASE_JD_REQ_CS, 0, 0, 0, 0);
   }
   uint8_t n2[8] = {1, 2, 3, 4, 21, 22, 23, 24};
   double t2 = now_s();
   int r = submit(&A, at, 8);
   int m2 = r ? 8 : wait_atoms(&A, n2, 8, 10.0);
   int ok2 = !r && !m2;
   for (unsigned i = 0; i < 8; i++)
      ok2 &= A.ev_code[n2[i]] == 1;
   for (unsigned i = 1; i <= 4; i++)
      ok2 &= A.tgt[100 + i] == 0x3000 + i && A.tgt[110 + i] == 0x3100 + i;
   say("A phase 2: %d missing after %.3f s; codes 1:0x%x 21:0x%x", m2, now_s() - t2, A.ev_code[1], A.ev_code[21]);
   check(ok2, "A after the fault: 8 more atoms DONE and written (context usable)");

   say("F1 phase 3: one more bad atom, then good atoms ORDER-dependent on it");
   clear_events(&A);
   atom(&at[0], 30, wv_job(&A, 40, bad, 6, 0x5555, 0), BASE_JD_REQ_CS, 0, 0, 0, 0);
   atom(&at[1], 31, write32_job(&A, 41, 131, 0x3131), BASE_JD_REQ_CS, 30, DEP_ORDER, 0, 0);
   uint8_t n3[2] = {30, 31};
   submit(&A, at, 2);
   int m3 = wait_atoms(&A, n3, 2, 15.0);
   say("A phase 3: %d missing; 30: 0x%x %s, 31: 0x%x %s tgt=0x%08x", m3, A.ev_code[30], code_name(A.ev_code[30]),
       A.ev_code[31], code_name(A.ev_code[31]), A.tgt[131]);
   check(!m3, "A: the second fault also reports");

   double tc = close_ctx(&A);
   say("A torn down in %.3f ms", tc * 1e3);
   close_ctx(&B);
}

/* ------------------------------------------------------------------------------------------------ f2 */
static void test_f2(int secs, const char *end)
{
   say("F2 never-ending, soft-stoppable job chain (J0 <-> J1 clearing each other), watch %d s, end=%s", secs, end);
   struct ctx A;
   ctx_open(&A, 'A');
   struct atom64 at[2];
   atom(&at[0], 1, loop_chain(&A, 0), BASE_JD_REQ_CS, 0, 0, 0, 0);
   double ts = now_s();
   check(submit(&A, at, 1) == 0, "loop atom submitted");
   for (int sec = 1; sec <= secs && !A.ev_count[1]; sec++) {
      read_events(&A, 1000);
      if (sec <= 2 || sec % 10 == 0)
         say("  t=%3d s: events for the loop atom: %d", sec, A.ev_count[1]);
   }
   say("after %.1f s: loop atom events %d (code 0x%x %s)", now_s() - ts, A.ev_count[1], A.ev_code[1],
       code_name(A.ev_code[1]));
   check(!A.ev_count[1], "no event while the chain loops (no hard-stop, no reset)");

   int ended = A.ev_count[1] != 0;
   if (!ended && !strcmp(end, "gpucut")) {
      /* FP_CUT_REQ: the cutter's core_req (default CS = slot 1; 0x1 = FS = slot 0); FP_CUT_PRIO: atom prio
       * (0 medium, 1 high, 2 low, 3 realtime). */
      atom(&at[0], 2, cut_job(&A, 4, 0), getenv("FP_CUT_REQ") ? strtoul(getenv("FP_CUT_REQ"), NULL, 0) : BASE_JD_REQ_CS,
           0, 0, 0, 0);
      at[0].prio = getenv("FP_CUT_PRIO") ? atoi(getenv("FP_CUT_PRIO")) : 0;
      say("cutter: core_req 0x%x prio %u", at[0].core_req, at[0].prio);
      double tu = now_s();
      submit(&A, at, 1);
      uint8_t n12[2] = {1, 2};
      int m = wait_atoms(&A, n12, 2, 10.0);
      say("GPU-side cut: %d of 2 atoms without an event after %.3f ms; cutter 0x%x %s at +%.3f ms, loop 0x%x %s at "
          "+%.3f ms", m, (now_s() - tu) * 1e3, A.ev_code[2], code_name(A.ev_code[2]),
          A.ev_count[2] ? (A.ev_t[2] - tu) * 1e3 : -1.0, A.ev_code[1], code_name(A.ev_code[1]),
          A.ev_count[1] ? (A.ev_t[1] - tu) * 1e3 : -1.0);
      check(!m && A.ev_code[1] == 1 && A.ev_code[2] == 1, "the cutter ran between soft-stops; the loop ended DONE");
      ended = A.ev_count[1] != 0;
      if (ended) {
         atom(&at[0], 3, write32_job(&A, 10, 0, 0x1234), BASE_JD_REQ_CS, 0, 0, 0, 0);
         submit(&A, at, 1);
         uint8_t three = 3;
         m = wait_atoms(&A, &three, 1, 5.0);
         check(!m && A.ev_code[3] == 1 && A.tgt[0] == 0x1234, "context usable after: a new atom DONE, written");
      }
   }
   if (!strcmp(end, "none")) {
      say("leaving the loop to process exit");
      return;
   }
   double tc = close_ctx(&A);
   say("context A torn down (loop %s): munmap+close took %.3f ms", ended ? "ended" : "STILL RUNNING", tc * 1e3);
}

/* ------------------------------------------------------------------------------------------------ f3 */
static void test_f3(int secs)
{
   say("F3 FENCE_WAIT on foreign fences that do not signal");
   struct ctx A, B;
   ctx_open(&A, 'A');
   ctx_open(&B, 'B');
   struct atom64 at[4];

   say("F3a control: B holds a fence with EVENT_WAIT (subject to soft_job_timeout); A waits on it");
   struct base_fence bf = { { -1, -1 } };
   B.evt[0] = 0;
   atom(&at[0], 1, (uint64_t)(uintptr_t)&B.evt[0], BASE_JD_REQ_SOFT_EVENT_WAIT, 0, 0, 0, 0);
   atom(&at[1], 2, (uint64_t)(uintptr_t)&bf, BASE_JD_REQ_SOFT_FENCE_TRIGGER, 1, DEP_DATA, 0, 0);
   submit(&B, at, 2);
   struct base_fence wf = { { bf.basep.fd, -1 } };
   atom(&at[0], 1, (uint64_t)(uintptr_t)&wf, BASE_JD_REQ_SOFT_FENCE_WAIT, 0, 0, 0, 0);
   atom(&at[1], 2, write32_job(&A, 0, 0, 0xA0A0), BASE_JD_REQ_CS, 1, DEP_DATA, 0, 0);
   double ta = now_s();
   submit(&A, at, 2);
   uint8_t n12[2] = {1, 2};
   int ma = wait_atoms(&A, n12, 2, 15.0);
   read_events(&B, 0);
   say("F3a: A %s after %.3f s: wait 0x%x %s, job 0x%x %s; B event-wait 0x%x %s, trigger 0x%x %s; fence status %d",
       ma ? "INCOMPLETE" : "complete", now_s() - ta, A.ev_code[1], code_name(A.ev_code[1]), A.ev_code[2],
       code_name(A.ev_code[2]), B.ev_code[1], code_name(B.ev_code[1]), B.ev_code[2], code_name(B.ev_code[2]),
       fence_status(bf.basep.fd));
   check(!ma, "control: the EVENT_WAIT-held fence resolves (by timeout) and A's atoms report");
   close(bf.basep.fd);
   clear_events(&A);
   clear_events(&B);

   say("F3b: B's never-ending chain -> FENCE_TRIGGER; A: FENCE_WAIT on it -> WRITE_VALUE; watch %d s", secs);
   struct base_fence lf = { { -1, -1 } };
   atom(&at[0], 10, loop_chain(&B, 0), BASE_JD_REQ_CS, 0, 0, 0, 0);
   atom(&at[1], 11, (uint64_t)(uintptr_t)&lf, BASE_JD_REQ_SOFT_FENCE_TRIGGER, 10, DEP_DATA, 0, 0);
   submit(&B, at, 2);
   say("B's fence fd %d", lf.basep.fd);
   struct base_fence wf2 = { { lf.basep.fd, -1 } };
   atom(&at[0], 10, (uint64_t)(uintptr_t)&wf2, BASE_JD_REQ_SOFT_FENCE_WAIT, 0, 0, 0, 0);
   atom(&at[1], 11, write32_job(&A, 1, 1, 0xB1B1), BASE_JD_REQ_CS, 10, DEP_DATA, 0, 0);
   double tb = now_s();
   submit(&A, at, 2);
   for (int sec = 1; sec <= secs; sec++) {
      read_events(&A, 1000);
      read_events(&B, 0);
      if (sec % 10 == 0 || A.ev_count[10] || A.ev_count[11])
         say("  t=%3d s: A events wait %d / job %d; B events loop %d / trigger %d; fence status %d", sec,
             A.ev_count[10], A.ev_count[11], B.ev_count[10], B.ev_count[11], fence_status(lf.basep.fd));
      if (A.ev_count[10] || A.ev_count[11])
         break;
   }
   say("F3b after %.1f s: A wait events %d, job events %d, tgt 0x%08x", now_s() - tb, A.ev_count[10],
       A.ev_count[11], A.tgt[1]);
   check(!A.ev_count[10] && !A.ev_count[11] && A.tgt[1] == 0xDEADBEEF,
         "FENCE_WAIT on a never-signalled foreign fence: no event, job not run");

   say("F3c: context C waits on the same fence and is torn down while waiting");
   struct ctx C;
   ctx_open(&C, 'C');
   struct base_fence wf3 = { { lf.basep.fd, -1 } };
   atom(&at[0], 1, (uint64_t)(uintptr_t)&wf3, BASE_JD_REQ_SOFT_FENCE_WAIT, 0, 0, 0, 0);
   atom(&at[1], 2, write32_job(&C, 0, 0, 0xC0C0), BASE_JD_REQ_CS, 1, DEP_DATA, 0, 0);
   submit(&C, at, 2);
   read_events(&C, 500);
   double tc = close_ctx(&C);
   say("C torn down with a pending FENCE_WAIT in %.3f ms", tc * 1e3);

   say("F3b end: the producer dies -- context B (loop + FENCE_TRIGGER) is torn down; A must then complete");
   double td = close_ctx(&B);
   say("B torn down in %.3f ms", td * 1e3);
   uint8_t n1011[2] = {10, 11};
   double tu = now_s();
   int ma2 = wait_atoms(&A, n1011, 2, 10.0);
   say("A after B's teardown (%.3f ms): wait 0x%x %s at +%.3f ms, job 0x%x %s, tgt 0x%08x; fence status %d",
       (now_s() - tu) * 1e3, A.ev_code[10], code_name(A.ev_code[10]),
       A.ev_count[10] ? (A.ev_t[10] - tu) * 1e3 : -1.0, A.ev_code[11], code_name(A.ev_code[11]), A.tgt[1],
       fence_status(lf.basep.fd));
   check(!ma2, "the producer's teardown resolves the fence: A's wait and job report");
   close(lf.basep.fd);
   close_ctx(&A);
}


/* ------------------------------------------------------------------------------------------------ f4 */
/* An infinite-loop compute shader (Valhall, assembled with Mesa's valhall/asm.py):
 *   MOV.i32 r2, u0.w0 ; MOV.i32 r3, u0.w1 ; IADD_IMM.i32 r0, 0x0, #0x0
 *   loop: IADD_IMM.i32.wait0 r0, r0, #0x1 ; STORE.i32.slot0 @r0, [r2:r3], offset:0
 *         BRANCHZ.reconverge 0x3F800000, offset:-3 (1.0 is never zero: always taken) ; NOP.end
 * It never finishes its workgroup, so soft-stop cannot take it off the GPU; each pass stores a counter to the
 * address in FAU word 0. */
static const uint64_t hang_code[] = {
   0x0091c20000000080ull, 0x0091c30000000081ull, 0x0110c000000000c0ull, 0x0910c00000000100ull,
   0x0061400218000002ull, 0x501fc007fffffdd0ull, 0x7800c00000000000ull,
};

struct hang {
   uint64_t code_va, target_va;
   volatile uint32_t *target;
   void *code;
};

/* Compute job in slot s; shader program descriptor in slot s+1, local storage in s+2, FAU in s+3. */
static uint64_t hang_job(struct ctx *c, unsigned s, struct hang *h)
{
   union kbase_ioctl_mem_alloc a = {0};
   a.in.va_pages = 1;
   a.in.commit_pages = 1;
   a.in.flags = BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_CPU_WR | BASE_MEM_PROT_GPU_RD | BASE_MEM_PROT_GPU_EX;
   if (ioctl(c->fd, KBASE_IOCTL_MEM_ALLOC, &a)) {
      say("exec MEM_ALLOC failed: %s", strerror(errno));
      exit(1);
   }
   h->code = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, c->fd, (off_t)a.out.gpu_va);
   if (h->code == MAP_FAILED) {
      say("exec mmap failed: %s", strerror(errno));
      exit(1);
   }
   h->code_va = a.out.gpu_va;
   memcpy(h->code, hang_code, sizeof(hang_code));
   c->extra[c->n_extra] = h->code;
   c->extra_sz[c->n_extra++] = 4096;
   h->target = alloc_map(c->fd, 1, RW, &h->target_va);
   h->target[0] = 0;

   uint32_t *prog = (uint32_t *)slot(c, s + 1), *tls = (uint32_t *)slot(c, s + 2);
   uint64_t *fau = (uint64_t *)slot(c, s + 3);
   memset(prog, 0, 128);
   prog[0] = 8u | (1u << 4) | (2u << 30);   /* type Shader, stage Compute, 32 registers per thread */
   put64(prog, 2, h->code_va);
   memset(tls, 0, 128);
   tls[1] = 31;                             /* WLS instances: log2(0x80000000) = no workgroup memory */
   memset(fau, 0, 128);
   fau[0] = h->target_va;

   uint32_t *d = (uint32_t *)slot(c, s);
   header(d, 4, 1, 0);                      /* Compute */
   d[8] = 0;                                /* workgroup 1x1x1 (minus 1) */
   d[9] = 1;                                /* task increment 1, axis X */
   d[10] = d[11] = d[12] = 1;               /* 1x1x1 workgroups */
   d[17] = 1;                               /* env: FAU count 1 (64-bit words) */
   put64(d, 26, slot_va(c, s + 1));         /* env: shader */
   put64(d, 28, slot_va(c, s + 2));         /* env: thread storage */
   put64(d, 30, slot_va(c, s + 3));         /* env: FAU */
   __sync_synchronize();
   say("hang job: code VA 0x%llx, store target VA 0x%llx", (unsigned long long)h->code_va,
       (unsigned long long)h->target_va);
   return slot_va(c, s);
}

/* Submit the hang; on its event, free the store target at once (what a driver does on a "terminal" event),
 * then reuse the context. The kernel side (stops, reset, MMU faults after the free) is in the faultrun trace. */
static void test_f4(int max_s, int with_fence)
{
   say("F4 infinite-loop compute shader (stores a counter every pass)%s; wait up to %d s",
       with_fence ? " + FENCE_TRIGGER on it" : "", max_s);
   struct ctx A;
   ctx_open(&A, 'A');
   struct hang h;
   struct atom64 at[2];
   struct base_fence bf = { { -1, -1 } };
   atom(&at[0], 1, hang_job(&A, 0, &h), BASE_JD_REQ_CS, 0, 0, 0, 0);
   atom(&at[1], 2, (uint64_t)(uintptr_t)&bf, BASE_JD_REQ_SOFT_FENCE_TRIGGER, 1, DEP_DATA, 0, 0);
   double ts = now_s();
   check(submit(&A, at, with_fence ? 2 : 1) == 0, "hang atom submitted");
   uint8_t one = 1;
   int m = 1;
   if (with_fence) {
      /* Watch the out-fence and the event fd together; whichever reports first, free the store target then. */
      int freed = 0;
      double tfence = 0;
      while (now_s() - ts < max_s && (!tfence || !A.ev_count[1])) {
         struct pollfd p[2] = { { bf.basep.fd, POLLIN, 0 }, { A.fd, POLLIN, 0 } };
         if (poll(p, 2, 1000) <= 0)
            continue;
         if ((p[0].revents & POLLIN) && !tfence) {
            tfence = now_s();
            say("FENCE signalled, status %d, %.3f s after submit", fence_status(bf.basep.fd), tfence - ts);
            if (!freed) { munmap((void *)h.target, 4096); freed = 1; say("store target freed (on the fence)"); }
         }
         if (p[1].revents & POLLIN)
            read_events(&A, 0);
         if (A.ev_count[1] && !freed) { munmap((void *)h.target, 4096); freed = 1; say("store target freed (on the event)"); }
      }
      say("fence at +%.3f s, hang event 0x%x %s at +%.3f s, trigger event 0x%x %s", tfence ? tfence - ts : -1.0,
          A.ev_code[1], code_name(A.ev_code[1]), A.ev_count[1] ? A.ev_t[1] - ts : -1.0, A.ev_code[2],
          code_name(A.ev_code[2]));
      if (tfence && A.ev_count[1])
         say("fence signalled %.3f ms %s the hang atom's event was read", fabs(A.ev_t[1] - tfence) * 1e3,
             tfence < A.ev_t[1] ? "BEFORE" : "after");
      usleep(1000000);
      read_events(&A, 0);
      check(A.ev_count[1] == 1 && A.ev_count[2] == 1, "both atoms reported exactly once");
      close(bf.basep.fd);
      double tc = close_ctx(&A);
      say("context torn down in %.3f ms", tc * 1e3);
      return;
   }
   for (int sec = 0; sec < max_s && m; sec++) {
      m = wait_atoms(&A, &one, 1, 1.0);
      if (m && (sec < 2 || sec % 5 == 4))
         say("  t=%2d s: no event; counter as the CPU sees it: %u", sec + 1, h.target[0]);
   }
   if (m) {
      say("no event in %d s", max_s);
   } else {
      double te = A.ev_t[1];
      munmap((void *)h.target, 4096);
      double tf = now_s();
      say("hang atom: code 0x%x %s after %.3f s; store target freed %.3f ms after the event", A.ev_code[1],
          code_name(A.ev_code[1]), te - ts, (tf - te) * 1e3);
   }
   usleep(2000000);
   read_events(&A, 0);
   check(A.ev_count[1] == 1, "the hang atom reported exactly once");
   if (!m) {
      atom(&at[0], 2, write32_job(&A, 20, 0, 0x4242), BASE_JD_REQ_CS, 0, 0, 0, 0);
      submit(&A, at, 1);
      uint8_t two = 2;
      int m2 = wait_atoms(&A, &two, 1, 10.0);
      say("reuse: atom 2 %s code 0x%x %s tgt 0x%08x", m2 ? "NO EVENT" : "event", A.ev_code[2],
          code_name(A.ev_code[2]), A.tgt[0]);
      check(!m2 && A.ev_code[2] == 1 && A.tgt[0] == 0x4242, "context usable after the hang");
   }
   double tc = close_ctx(&A);
   say("context torn down in %.3f ms", tc * 1e3);
}

/* f5: the P0 barrier premise. Hang atom 1; barrier atom 2 = WRITE_VALUE ORDER-dependent on 1 (a failure does not
 * cancel it, so it runs after the hang ends, whatever its code); atom 3 = FENCE_TRIGGER DATA-dependent on 2. Under a
 * forced reset the kernel may resolve atom 1's dependents during the reset (f4f6): does kbase still put the barrier
 * on the GPU only after the reset, and does the fence then signal after it? The trace answers (barrier pull vs
 * kbase_pm_init_hw's return). */
static void test_f5(int max_s)
{
   say("F5 barrier after a hang: hang(1) -> ORDER -> WRITE_VALUE barrier(2) -> DATA -> FENCE_TRIGGER(3)");
   struct ctx A;
   ctx_open(&A, 'A');
   struct hang h;
   struct atom64 at[3];
   struct base_fence bf = { { -1, -1 } };
   A.tgt[0] = 0xDEADBEEF;
   atom(&at[0], 1, hang_job(&A, 0, &h), BASE_JD_REQ_CS, 0, 0, 0, 0);
   atom(&at[1], 2, write32_job(&A, 8, 0, 0xBA77), BASE_JD_REQ_CS, 1, DEP_ORDER, 0, 0);
   atom(&at[2], 3, (uint64_t)(uintptr_t)&bf, BASE_JD_REQ_SOFT_FENCE_TRIGGER, 2, DEP_DATA, 0, 0);
   double ts = now_s();
   check(submit(&A, at, 3) == 0, "hang + barrier + trigger submitted");
   double tfence = 0;
   uint8_t all[3] = {1, 2, 3};
   while (now_s() - ts < max_s) {
      struct pollfd p[2] = { { bf.basep.fd, POLLIN, 0 }, { A.fd, POLLIN, 0 } };
      if (poll(p, 2, 1000) > 0) {
         if ((p[0].revents & POLLIN) && !tfence) {
            tfence = now_s();
            say("FENCE signalled, status %d", fence_status(bf.basep.fd));
         }
         if (p[1].revents & POLLIN)
            read_events(&A, 0);
      }
      if (tfence && !wait_atoms(&A, all, 3, 0.0))
         break;
   }
   say("hang 0x%x %s +%.3f s; barrier 0x%x %s +%.3f s tgt 0x%08x; trigger 0x%x %s; fence +%.3f s", A.ev_code[1],
       code_name(A.ev_code[1]), A.ev_count[1] ? A.ev_t[1] - ts : -1.0, A.ev_code[2], code_name(A.ev_code[2]),
       A.ev_count[2] ? A.ev_t[2] - ts : -1.0, A.tgt[0], A.ev_code[3], code_name(A.ev_code[3]),
       tfence ? tfence - ts : -1.0);
   check(A.ev_code[2] == 1 && A.tgt[0] == 0xBA77, "the barrier ran and reported DONE");
   check(tfence && fence_status(bf.basep.fd) == 1, "the fence behind the barrier signalled without error");
   close(bf.basep.fd);
   close_ctx(&A);
}

/* Isolation: one WRITE_VALUE of <type> to ts[0], optionally jumping to itself; report the first event. */
static void test_wv(unsigned type, int loop)
{
   struct ctx A;
   ctx_open(&A, 'A');
   struct atom64 at[1];
   uint64_t jc = wv_job(&A, 0, (uint64_t)(uintptr_t)&A.ts[0], type, 0x77, 0);
   if (loop)
      put64((uint32_t *)slot(&A, 0), 6, jc);
   __sync_synchronize();
   atom(&at[0], 1, jc, BASE_JD_REQ_CS, 0, 0, 0, 0);
   submit(&A, at, 1);
   uint8_t one = 1;
   int m = wait_atoms(&A, &one, 1, 2.0);
   uint32_t *d = (uint32_t *)slot(&A, 0);
   say("wv type %u loop %d: %s code 0x%x %s; ts[0]=0x%llx; header w0 (exception status) 0x%08x w1 0x%08x",
       type, loop, m ? "no event in 2 s" : "event", A.ev_code[1], code_name(A.ev_code[1]),
       (unsigned long long)A.ts[0], d[0], d[1]);
   double tc = close_ctx(&A);
   say("torn down in %.3f ms", tc * 1e3);
}

int main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IOLBF, 0);
   t0_s = now_s();
   marker_fd = open("/sys/kernel/tracing/trace_marker", O_WRONLY | O_CLOEXEC);
   const char *t = argc > 1 ? argv[1] : "";
   if (!strcmp(t, "f1"))
      test_f1(argc > 2 ? argv[2] : "tf");
   else if (!strcmp(t, "f2"))
      test_f2(argc > 2 ? atoi(argv[2]) : 20, argc > 3 ? argv[3] : "gpucut");
   else if (!strcmp(t, "f4"))
      test_f4(argc > 2 ? atoi(argv[2]) : 30, argc > 3 && !strcmp(argv[3], "fence"));
   else if (!strcmp(t, "f5"))
      test_f5(argc > 2 ? atoi(argv[2]) : 30);
   else if (!strcmp(t, "wv"))
      test_wv(argc > 2 ? atoi(argv[2]) : 6, argc > 3 ? atoi(argv[3]) : 0);
   else if (!strcmp(t, "f3"))
      test_f3(argc > 2 ? atoi(argv[2]) : 60);
   else {
      printf("usage: faultprobe f1 [tf|pf] | f2 <secs> <gpucut|close|none> | f3 <secs> | f4 <max secs> [fence] | f5 <max secs> | wv <type> <loop>\n");
      return 2;
   }
   say("RESULT %s (%d failed checks)", fails ? "FAIL" : "PASS", fails);
   return fails ? 1 : 0;
}
