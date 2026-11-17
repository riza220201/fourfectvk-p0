/*
 * Copyright 2026 Noysz
 * SPDX-License-Identifier: MIT
 */

#include "panvk_mem_prof.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "util/os_time.h"
#include "util/u_atomic.h"

#include "panfrost/lib/pan_env.h"
#include "panvk_device.h"
#include "panvk_physical_device.h"
#include "panvk_priv_bo.h"

#ifdef HAVE_PAN_KMOD_KBASE
#include "kmod/kbase_kmod.h"
#endif

#define MB(x) ((double)(x) / (1024.0 * 1024.0))

void
panvk_mem_prof_init(struct panvk_device *dev)
{
   struct panvk_mem_prof *p = &dev->mem_prof;
   const char *v = pan_env_flag("PANVK_MEM_PROF");

   memset(p, 0, sizeof(*p));
   if (!v)
      return;

   p->enabled = true;
   p->out = stderr;
   if (strchr(v, '/')) {
      FILE *f = fopen(v, "a");
      if (f) {
         setvbuf(f, NULL, _IOLBF, 0);
         p->out = f;
      }
   }
   p->start_ns = os_time_get_nano();
   p->last_ns = 0;
   fprintf(p->out, "panvk-mem: profiling on (pid %d)\n", (int)getpid());
}

void
panvk_mem_prof_finish(struct panvk_device *dev)
{
   struct panvk_mem_prof *p = &dev->mem_prof;

   if (!p->enabled)
      return;

   p->last_ns = 0;
   panvk_mem_prof_tick(dev);
   if (p->out && p->out != stderr)
      fclose(p->out);
   p->enabled = false;
}

/* "Key:   1234 kB" from a /proc file, in bytes; 0 if missing. */
static uint64_t
proc_kb(const char *buf, const char *key)
{
   const char *s = buf ? strstr(buf, key) : NULL;

   if (!s)
      return 0;

   return strtoull(s + strlen(key), NULL, 10) * 1024;
}

static char *
read_small_file(const char *path, char *buf, size_t size)
{
   FILE *f = fopen(path, "r");

   if (!f)
      return NULL;

   size_t n = fread(buf, 1, size - 1, f);
   fclose(f);
   buf[n] = 0;
   return buf;
}

static uint64_t
commit_of(struct panvk_priv_bo *bo)
{
#ifdef HAVE_PAN_KMOD_KBASE
   if (bo && bo->bo)
      return kbase_kmod_bo_commit_bytes(bo->bo);
#endif
   return 0;
}

void
panvk_mem_prof_tick(struct panvk_device *dev)
{
   struct panvk_mem_prof *p = &dev->mem_prof;

   if (!p->enabled)
      return;

   uint64_t now = os_time_get_nano();
   if (p->last_ns && now - p->last_ns < 2000000000ull)
      return;
   p->last_ns = now;

   struct panvk_physical_device *pdev =
      to_panvk_physical_device(dev->vk.physical);
   uint64_t devmem = p_atomic_read(&pdev->memory.heap_used);
   uint64_t cached = p_atomic_read(&dev->pool_cached_bytes);

   uint64_t bos = 0, va = 0, commit = 0, growable = 0, allocs = 0, frees = 0;
   uint64_t grow_commit = 0, lazy_n = 0, lazy_va = 0, lazy_commit = 0;
#ifdef HAVE_PAN_KMOD_KBASE
   struct kbase_kmod_mem_stats ks;
   kbase_kmod_mem_stats(&ks);
   bos = ks.live_bos;
   va = ks.live_bytes;
   commit = ks.commit_bytes;
   growable = ks.growable_bos;
   allocs = ks.allocs;
   frees = ks.frees;
   grow_commit = ks.growable_commit_bytes;
   lazy_n = ks.lazy_bos;
   lazy_va = ks.lazy_va_bytes;
   lazy_commit = ks.lazy_commit_bytes;
#endif

   char buf[4096];
   char *st = read_small_file("/proc/self/status", buf, sizeof(buf));
   uint64_t rss = proc_kb(st, "VmRSS:"), anon = proc_kb(st, "RssAnon:"),
            file = proc_kb(st, "RssFile:"), shmem = proc_kb(st, "RssShmem:");
   char *mi = read_small_file("/proc/meminfo", buf, sizeof(buf));
   uint64_t avail = proc_kb(mi, "MemAvailable:");

   fprintf(p->out,
           "panvk-mem t=%.1fs | kbase BOs %" PRIu64 " (%.1f MB VA, %.1f MB "
           "committed at alloc, %" PRIu64 " growable) +%" PRIu64 "/-%" PRIu64
           " | VkDeviceMemory %.1f MB (lazy: %" PRIu64 " BOs, %.1f MB of "
           "%.1f MB backed) | all growable backed %.1f MB | cmd pool cache "
           "%.1f MB | heaps "
           "committed: tiler %.1f, poly %.1f, indirect varyings %.1f MB | "
           "RSS %.1f MB (anon %.1f, file %.1f, shmem %.1f) | MemAvailable "
           "%.1f MB\n",
           (now - p->start_ns) / 1e9, bos, MB(va), MB(commit), growable,
           allocs - p->last_allocs, frees - p->last_frees, MB(devmem),
           lazy_n, MB(lazy_commit), MB(lazy_va), MB(grow_commit), MB(cached), MB(commit_of(dev->tiler_heap)),
           MB(commit_of(dev->poly_heap_data)),
           MB(commit_of(dev->indirect_varying_buffer)), MB(rss), MB(anon),
           MB(file), MB(shmem), MB(avail));

   p->last_allocs = allocs;
   p->last_frees = frees;
}
