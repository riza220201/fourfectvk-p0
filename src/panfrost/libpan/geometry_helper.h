/*
 * Copyright 2026 panvk-g99-jm contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdint.h>
#include "compiler/libcl/libcl.h"

/* Compute jobs of a geometry shader draw whose grid comes from the GPU. */
enum panlib_gs_job {
   PANLIB_GS_JOB_VS = 0,
   PANLIB_GS_JOB_COUNT = 1,
   PANLIB_GS_JOB_MAIN = 2,
   PANLIB_GS_JOB_XFB = 3,
   PANLIB_GS_JOBS = 4,
};

/* Parameters of panlib_gs_setup_indirect(), written by the CPU per draw. */
struct panlib_gs_indirect_params {
   /* VkDrawIndirectCommand or VkDrawIndexedIndirectCommand. */
   uint64_t draw;
   /* u32 draw count of vkCmdDraw*IndirectCount, 0 otherwise. */
   uint64_t draw_count;
   uint64_t index_buffer;
   /* struct poly_vertex_params, poly_geometry_params, poly_heap. */
   uint64_t vp;
   uint64_t p;
   uint64_t heap;
   /* Output mask of the stage before the GS (VS or TES). */
   uint64_t vs_outputs;
   /* COMPUTE jobs parked as NOT_STARTED, 0 when not queued. */
   uint64_t jobs[PANLIB_GS_JOBS];
   /* gl_BaseVertex / gl_BaseInstance sysvals of the software VS push
    * uniforms, 0 when unused. */
   uint64_t first_vertex_sysval;
   uint64_t base_instance_sysval;
   uint32_t index_size_B;
   uint32_t index_buffer_range_el;
   uint32_t prim;
   uint32_t prefix_sum;
   uint32_t max_indices;
   uint32_t shape;
   uint32_t draw_index;
   uint32_t pad;
};

/* Parameters of panlib_unroll_restart(), written by the CPU per draw. */
struct panlib_unroll_restart_params {
   /* struct poly_heap, the new index buffer is allocated from it. */
   uint64_t heap;
   /* Bound index buffer (without the first index) and its size. */
   uint64_t index_buffer;
   /* VkDrawIndexedIndirectCommand in, and the one the GS draw reads. */
   uint64_t in_draw;
   uint64_t out_draw;
   /* u32 draw count of vkCmdDrawIndexedIndirectCount, 0 otherwise. */
   uint64_t draw_count;
   uint32_t index_buffer_range_el;
   uint32_t index_size_B;
   uint32_t restart_index;
   uint32_t flatshade_first;
   uint32_t prim;
   uint32_t draw_index;
};

/* Parameters of panlib_stats_add(): one draw or dispatch inside a pipeline
 * statistics query. counter[] holds the address of each statistic, indexed
 * by its VkQueryPipelineStatisticFlagBits bit, 0 when not counted. */
#define PANLIB_STATS 11
struct panlib_stats_params {
   uint64_t counter[PANLIB_STATS];
   /* Indirect command (draw, indexed draw or dispatch), 0 when direct. */
   uint64_t cmd;
   /* u32 draw count of vkCmdDraw*IndirectCount, 0 otherwise. */
   uint64_t draw_count;
   /* Indexed indirect draw the tessellator wrote (index count, instances),
    * and the draw a GS wrote, 0 when there is none or it is CPU-known. */
   uint64_t tess_draw;
   uint64_t gs_draw;
   uint64_t cs_invocations;
   uint32_t vertex_count;
   uint32_t instance_count;
   uint32_t prim;
   uint32_t patch_size;
   uint32_t tess_verts_per_prim;
   uint32_t gs_invocations;
   uint32_t gs_count;
   uint32_t gs_instances;
   uint32_t cs_local_size;
   uint32_t compute;
   uint32_t draw_index;
   uint32_t pad;
};

/* Compute jobs of a tessellated draw whose grid comes from the GPU. */
enum panlib_tess_job {
   PANLIB_TESS_JOB_VS = 0,
   PANLIB_TESS_JOB_TCS = 1,
   PANLIB_TESS_JOBS = 2,
};

/* Parameters of panlib_tess_setup_indirect(), written by the CPU per draw. */
struct panlib_tess_indirect_params {
   /* VkDrawIndirectCommand or VkDrawIndexedIndirectCommand. */
   uint64_t draw;
   /* u32 draw count of vkCmdDraw*IndirectCount, 0 otherwise. */
   uint64_t draw_count;
   uint64_t index_buffer;
   /* struct poly_tess_params, poly_vertex_params. */
   uint64_t tess;
   uint64_t vp;
   uint64_t vs_outputs;
   uint64_t jobs[PANLIB_TESS_JOBS];
   uint64_t first_vertex_sysval;
   uint64_t base_instance_sysval;
   uint32_t index_size_B;
   uint32_t index_buffer_range_el;
   uint32_t draw_index;
   uint32_t pad;
};
