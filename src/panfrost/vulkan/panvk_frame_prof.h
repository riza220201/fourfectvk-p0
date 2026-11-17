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
#define PANVK_FRAME_PROF_GROUPS 32
#define PANVK_FRAME_PROF_JOB_TYPES 16

/* What one render pass (JM batch) renders. Filled when the batch is closed,
 * only while PANVK_FRAME_PROF_PASSES=1. Passes with the same key are added
 * up in one group. */
struct panvk_frame_prof_pass {
   uint16_t width, height;  /* framebuffer, 0x0 for compute-only batches */
   uint16_t tile_px;        /* pixels per tile */
   uint8_t samples, rts, zs;
   uint8_t loads;           /* attachments read back from memory first */
   uint8_t stores;          /* attachments written out at the end */
   uint8_t afbc;            /* stored colour targets that are AFBC */
   uint8_t cont;            /* continues a render pass split into batches */
};

struct panvk_frame_prof_group {
   struct panvk_frame_prof_pass key;
   uint32_t count;
   uint32_t jobs[PANVK_FRAME_PROF_JOB_TYPES];
   uint64_t vtc_ns, frag_ns;
};

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

   /* PANVK_FRAME_PROF_PASSES=1: per-pass groups and job counts by type.
    * Times per group need PANVK_KBASE_SYNC_SUBMIT=1. Under mtx. */
   bool passes;
   struct panvk_frame_prof_group groups[PANVK_FRAME_PROF_GROUPS];
   unsigned ngroups, groups_dropped;
   uint64_t jobs[PANVK_FRAME_PROF_JOB_TYPES];
};

/* One submitted batch: its key, job counts by job header type, and the wall
 * time of its two chains (0 when not measured). */
void panvk_frame_prof_pass(struct panvk_device *dev,
                           const struct panvk_frame_prof_pass *pass,
                           const uint32_t *jobs, uint64_t vtc_ns,
                           uint64_t frag_ns);

void panvk_frame_prof_init(struct panvk_device *dev);
void panvk_frame_prof_finish(struct panvk_device *dev);

/* A present that took [start_ns, end_ns] on the calling thread. */
void panvk_frame_prof_present(struct panvk_device *dev, uint64_t start_ns,
                              uint64_t end_ns);

/* Prints the window if 2 s passed (also called from submits, so a log keeps
 * coming when presents are not seen). */
void panvk_frame_prof_tick(struct panvk_device *dev, uint64_t now_ns);

#endif /* PANVK_FRAME_PROF_H */
