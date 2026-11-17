/*
 * Copyright 2026 Noysz
 * SPDX-License-Identifier: MIT
 */

/* PANVK_FRAME_PROF=1 (stderr) or =<file>: every 2 s, frame pacing and where
 * the time goes. Frames are counted at vkQueuePresentKHR. GPU busy is the
 * union of [submit, completion seen] of every kbase submission, so it can
 * read a little high when nobody waits for a while. Off by default; when
 * off every hook is one predictable branch. */

#ifndef PANVK_FRAME_PROF_H
#define PANVK_FRAME_PROF_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "c11/threads.h"

struct panvk_device;

#define PANVK_FRAME_PROF_MAX_FRAMES 4096

struct panvk_frame_prof {
   bool enabled;
   FILE *out;
   mtx_t mtx;
   uint64_t start_ns, window_ns, last_present_ns;
   uint64_t cpu_ticks; /* process utime + stime at window start */

   /* Present intervals of the current window, in us. */
   uint32_t frame_us[PANVK_FRAME_PROF_MAX_FRAMES];
   unsigned nframes;

   /* Added from the queue and WSI hooks (atomics). */
   uint64_t gpu_busy_ns, submit_ns, wait_ns, present_ns;
   uint64_t submits, batches, waits;
   /* PANVK_KBASE_SYNC_SUBMIT=1 only (one chain at a time, each waited
    * for): wall time of the vertex/tiler/compute and fragment chains. */
   uint64_t vtc_ns, frag_ns;
};

void panvk_frame_prof_init(struct panvk_device *dev);
void panvk_frame_prof_finish(struct panvk_device *dev);

/* A present that took [start_ns, end_ns] on the calling thread. */
void panvk_frame_prof_present(struct panvk_device *dev, uint64_t start_ns,
                              uint64_t end_ns);

/* Prints the window if 2 s passed (also called from submits, so a log keeps
 * coming when presents are not seen). */
void panvk_frame_prof_tick(struct panvk_device *dev, uint64_t now_ns);

#endif /* PANVK_FRAME_PROF_H */
