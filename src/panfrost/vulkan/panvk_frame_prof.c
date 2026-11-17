/*
 * Copyright 2026 Noysz
 * SPDX-License-Identifier: MIT
 */

#include "panvk_frame_prof.h"

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "util/os_time.h"
#include "util/u_atomic.h"

#include "panfrost/lib/pan_env.h"
#include "panvk_device.h"

#define WINDOW_NS 2000000000ull

/* utime + stime of the whole process (every thread), in clock ticks. */
static uint64_t
process_cpu_ticks(void)
{
   char buf[1024];
   FILE *f = fopen("/proc/self/stat", "r");
   if (!f)
      return 0;
   size_t n = fread(buf, 1, sizeof(buf) - 1, f);
   fclose(f);
   buf[n] = 0;

   /* The command name may contain spaces: fields start after the last ')'. */
   const char *s = strrchr(buf, ')');
   if (!s)
      return 0;
   unsigned long long utime = 0, stime = 0;
   if (sscanf(s + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu",
              &utime, &stime) != 2)
      return 0;
   return utime + stime;
}

void
panvk_frame_prof_init(struct panvk_device *dev)
{
   struct panvk_frame_prof *p = &dev->frame_prof;
   const char *v = pan_env_flag("PANVK_FRAME_PROF");

   memset(p, 0, sizeof(*p));
   if (!v)
      return;
   if (mtx_init(&p->mtx, mtx_plain) != thrd_success)
      return;

   p->out = stderr;
   if (strchr(v, '/')) {
      FILE *f = fopen(v, "a");
      if (f) {
         setvbuf(f, NULL, _IOLBF, 0);
         p->out = f;
      }
   }
   p->start_ns = p->window_ns = os_time_get_nano();
   p->cpu_ticks = process_cpu_ticks();
   p->passes = pan_env_flag("PANVK_FRAME_PROF_PASSES") != NULL;
   p->enabled = true;
   fprintf(p->out, "panvk-frame: profiling on (pid %d), one line per 2 s%s\n",
           (int)getpid(), p->passes ? ", with render pass groups" : "");
}

void
panvk_frame_prof_finish(struct panvk_device *dev)
{
   struct panvk_frame_prof *p = &dev->frame_prof;

   if (!p->enabled)
      return;
   p->enabled = false;
   if (p->out && p->out != stderr)
      fclose(p->out);
   mtx_destroy(&p->mtx);
}

static int
cmp_u32(const void *a, const void *b)
{
   uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
   return x < y ? -1 : x > y;
}

void
panvk_frame_prof_pass(struct panvk_device *dev,
                      const struct panvk_frame_prof_pass *pass,
                      const uint32_t *jobs, uint64_t vtc_ns, uint64_t frag_ns)
{
   struct panvk_frame_prof *p = &dev->frame_prof;

   if (!p->enabled || !p->passes)
      return;

   mtx_lock(&p->mtx);
   for (unsigned t = 0; t < PANVK_FRAME_PROF_JOB_TYPES; t++)
      p->jobs[t] += jobs[t];

   struct panvk_frame_prof_group *g = NULL;
   for (unsigned i = 0; i < p->ngroups; i++) {
      if (!memcmp(&p->groups[i].key, pass, sizeof(*pass))) {
         g = &p->groups[i];
         break;
      }
   }
   if (!g && p->ngroups < PANVK_FRAME_PROF_GROUPS) {
      g = &p->groups[p->ngroups++];
      memset(g, 0, sizeof(*g));
      g->key = *pass;
   }
   if (g) {
      g->count++;
      for (unsigned t = 0; t < PANVK_FRAME_PROF_JOB_TYPES; t++)
         g->jobs[t] += jobs[t];
      g->vtc_ns += vtc_ns;
      g->frag_ns += frag_ns;
   } else {
      p->groups_dropped++;
   }
   mtx_unlock(&p->mtx);
}

/* Job header types (genxml), for the job count line. */
static const char *const job_type_names[PANVK_FRAME_PROF_JOB_TYPES] = {
   [1] = "null",       [2] = "write value", [3] = "cache flush",
   [4] = "compute",    [5] = "vertex",      [6] = "geometry",
   [7] = "tiler",      [8] = "fused",       [9] = "fragment",
   [10] = "indexed vertex", [11] = "malloc vertex",
};

static uint64_t
group_cost(const struct panvk_frame_prof_group *g)
{
   /* Without chain times (async submit), rank by how often it ran. */
   return g->vtc_ns + g->frag_ns ? g->vtc_ns + g->frag_ns : g->count;
}

static int
cmp_group(const void *a, const void *b)
{
   uint64_t x = group_cost(a), y = group_cost(b);
   return x > y ? -1 : x < y;
}

/* Caller holds p->mtx. Prints and resets the pass groups of a window. */
static void
print_passes(struct panvk_frame_prof *p, double win_s, double per,
             const char *unit)
{
   char line[512];
   int len = snprintf(line, sizeof(line), "panvk-frame   jobs per %s:", unit);
   for (unsigned t = 0; t < PANVK_FRAME_PROF_JOB_TYPES; t++) {
      if (!p->jobs[t] || len >= (int)sizeof(line))
         continue;
      len += snprintf(line + len, sizeof(line) - len, " %s %.1f,",
                      job_type_names[t] ? job_type_names[t] : "other",
                      p->jobs[t] / per);
   }
   if (len > 0 && len < (int)sizeof(line) && line[len - 1] == ',')
      line[len - 1] = 0;
   fprintf(p->out, "%s\n", line);

   uint64_t total = 0;
   for (unsigned i = 0; i < p->ngroups; i++)
      total += p->groups[i].vtc_ns + p->groups[i].frag_ns;

   qsort(p->groups, p->ngroups, sizeof(p->groups[0]), cmp_group);
   for (unsigned i = 0; i < p->ngroups && i < 10; i++) {
      const struct panvk_frame_prof_group *g = &p->groups[i];
      const struct panvk_frame_prof_pass *k = &g->key;
      const uint32_t draws = g->jobs[7] + g->jobs[10] + g->jobs[11];
      char time[160] = "";
      if (total)
         snprintf(time, sizeof(time),
                  " | %.0f%% of GPU time: vertex/tiler/compute %.1f ms, "
                  "fragment %.1f ms per %s",
                  100.0 * (g->vtc_ns + g->frag_ns) / total,
                  g->vtc_ns / 1e6 / per, g->frag_ns / 1e6 / per, unit);
      if (k->width)
         fprintf(p->out,
                 "panvk-frame   pass %ux%u %urt%s %ux tile %upx load %u "
                 "store %u afbc %u%s: %.1f per %s, %.0f draws, %.0f compute "
                 "jobs each%s\n",
                 k->width, k->height, k->rts, k->zs ? "+zs" : "", k->samples,
                 k->tile_px, k->loads, k->stores, k->afbc,
                 k->cont ? " (split)" : "", g->count / per, unit,
                 (double)draws / g->count, (double)g->jobs[4] / g->count,
                 time);
      else
         fprintf(p->out,
                 "panvk-frame   no framebuffer: %.1f per %s, %.0f compute "
                 "jobs each%s\n",
                 g->count / per, unit, (double)g->jobs[4] / g->count, time);
   }
   if (p->ngroups > 10 || p->groups_dropped)
      fprintf(p->out, "panvk-frame   (%u more groups, %u passes not grouped)\n",
              p->ngroups > 10 ? p->ngroups - 10 : 0, p->groups_dropped);

   (void)win_s;
   p->ngroups = 0;
   p->groups_dropped = 0;
   memset(p->jobs, 0, sizeof(p->jobs));
}

void
panvk_frame_prof_tick(struct panvk_device *dev, uint64_t now)
{
   struct panvk_frame_prof *p = &dev->frame_prof;

   if (!p->enabled || now - p->window_ns < WINDOW_NS)
      return;

   mtx_lock(&p->mtx);
   if (now - p->window_ns < WINDOW_NS) {
      mtx_unlock(&p->mtx);
      return;
   }

   const double win_s = (now - p->window_ns) / 1e9;
   const uint64_t gpu = p_atomic_xchg(&p->gpu_busy_ns, 0);
   const uint64_t submit = p_atomic_xchg(&p->submit_ns, 0);
   const uint64_t wait = p_atomic_xchg(&p->wait_ns, 0);
   const uint64_t present = p_atomic_xchg(&p->present_ns, 0);
   const uint64_t submits = p_atomic_xchg(&p->submits, 0);
   const uint64_t batches = p_atomic_xchg(&p->batches, 0);
   const uint64_t vtc = p_atomic_xchg(&p->vtc_ns, 0);
   const uint64_t frag = p_atomic_xchg(&p->frag_ns, 0);
   char split[96] = "";
   if (vtc + frag)
      snprintf(split, sizeof(split),
               " | chains: vertex/tiler/compute %.0f%%, fragment %.0f%%",
               100.0 * vtc / (vtc + frag), 100.0 * frag / (vtc + frag));
   const uint64_t ticks = process_cpu_ticks();
   const long hz = sysconf(_SC_CLK_TCK);
   const double cpu_pct =
      hz > 0 ? 100.0 * (ticks - p->cpu_ticks) / hz / win_s : 0;
   const unsigned n = p->nframes;
   /* Per frame when frames were seen, else per second. */
   const double per = n ? n : win_s;
   const char *unit = n ? "frame" : "s";

   if (n) {
      double sum = 0, sq = 0;
      for (unsigned i = 0; i < n; i++)
         sum += p->frame_us[i];
      const double mean = sum / n;
      for (unsigned i = 0; i < n; i++)
         sq += (p->frame_us[i] - mean) * (p->frame_us[i] - mean);
      qsort(p->frame_us, n, sizeof(p->frame_us[0]), cmp_u32);
      const uint32_t p99 = p->frame_us[(n * 99) / 100 < n ? (n * 99) / 100
                                                          : n - 1];
      fprintf(p->out,
              "panvk-frame t=%.1fs | %.1f fps, 1%% low %.1f, worst %.0f ms, "
              "frame time %.1f +- %.1f ms | GPU busy %.0f%% | process CPU "
              "%.0f%% | per frame: submit %.2f ms, wait for GPU %.2f ms, "
              "present %.2f ms, %.1f submits, %.1f passes%s\n",
              (now - p->start_ns) / 1e9, n / win_s,
              p99 ? 1e6 / p99 : 0, p->frame_us[n - 1] / 1000.0,
              mean / 1000.0, sqrt(sq / n) / 1000.0,
              100.0 * gpu / (now - p->window_ns), cpu_pct,
              submit / 1e6 / per, wait / 1e6 / per, present / 1e6 / per,
              submits / per, batches / per, split);
   } else {
      fprintf(p->out,
              "panvk-frame t=%.1fs | no presents seen | GPU busy %.0f%% | "
              "process CPU %.0f%% | per %s: submit %.2f ms, wait for GPU "
              "%.2f ms, %.1f submits, %.1f passes%s\n",
              (now - p->start_ns) / 1e9, 100.0 * gpu / (now - p->window_ns),
              cpu_pct, unit, submit / 1e6 / per, wait / 1e6 / per,
              submits / per, batches / per, split);
   }

   if (p->passes)
      print_passes(p, win_s, per, unit);

   p->nframes = 0;
   p->window_ns = now;
   p->cpu_ticks = ticks;
   mtx_unlock(&p->mtx);
}

void
panvk_frame_prof_present(struct panvk_device *dev, uint64_t start_ns,
                         uint64_t end_ns)
{
   struct panvk_frame_prof *p = &dev->frame_prof;

   if (!p->enabled)
      return;

   mtx_lock(&p->mtx);
   if (p->last_present_ns && p->nframes < PANVK_FRAME_PROF_MAX_FRAMES) {
      uint64_t us = (start_ns - p->last_present_ns) / 1000;
      p->frame_us[p->nframes++] = us > UINT32_MAX ? UINT32_MAX : us;
   }
   p->last_present_ns = start_ns;
   mtx_unlock(&p->mtx);

   p_atomic_add(&p->present_ns, end_ns - start_ns);
   panvk_frame_prof_tick(dev, end_ns);
}
