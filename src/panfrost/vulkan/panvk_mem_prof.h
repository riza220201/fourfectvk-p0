/*
 * Copyright 2026 Noysz
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_MEM_PROF_H
#define PANVK_MEM_PROF_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

struct panvk_device;

/* PANVK_MEM_PROF=1 (stderr) or =<file>: every 2 s, memory held by the driver
 * (kbase BOs, VkDeviceMemory, cached command pool BOs, growable heaps) and
 * the process/system memory state. */
struct panvk_mem_prof {
   bool enabled;
   FILE *out;
   uint64_t start_ns, last_ns;
   uint64_t last_allocs, last_frees;
};

void panvk_mem_prof_init(struct panvk_device *dev);
void panvk_mem_prof_finish(struct panvk_device *dev);
void panvk_mem_prof_tick(struct panvk_device *dev);

#endif /* PANVK_MEM_PROF_H */
