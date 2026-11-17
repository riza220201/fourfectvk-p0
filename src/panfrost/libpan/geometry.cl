/*
 * Copyright 2026 panvk-g99-jm contributors
 * SPDX-License-Identifier: MIT
 */

#include "compiler/libcl/libcl.h"
#include "compiler/libcl/libcl_vk.h"
#include "poly/geometry.h"
#include "poly/prim.h"
#include "genxml/gen_macros.h"
#include "geometry_helper.h"
#include "poly/tessellator.h"

#if PAN_ARCH == 9
/* PATCH v9: transform feedback from a geometry shader with a dynamic vertex
 * count needs an inclusive prefix sum over the per-primitive counts written
 * by the GS COUNT pass. libagx uses a 1024-wide workgroup scan, which Mali-G57
 * cannot run (max 512 invocations, subgroup 16, see the tessellation prefix
 * sum). Single-thread serial version with the same result. */
KERNEL(1)
panlib_prefix_sum_geom(constant struct poly_geometry_params *p)
{
   const uint words = p->count_buffer_stride / 4;
   global uint *buf = p->count_buffer;

   for (uint w = 0; w < words; w++) {
      uint sum = 0;
      for (uint i = 0; i < p->input_primitives; i++) {
         sum += buf[(i * words) + w];
         buf[(i * words) + w] = sum;
      }
   }
}

/* Turn a parked COMPUTE job into a real one with the given grid, or into a
 * NULL job when the grid is empty (same as panlib_indirect_dispatch). */
static void
panlib_gs_patch_job(uint64_t job_addr, uint32_t x, uint32_t y)
{
   if (!job_addr)
      return;

   global uint8_t *job = (global uint8_t *)job_addr;
   const bool is_no_op = x == 0 || y == 0;

   struct mali_job_header_packed job_header;
   pan_pack(&job_header, JOB_HEADER, cfg) {
      cfg.type = is_no_op ? MALI_JOB_TYPE_NULL : MALI_JOB_TYPE_COMPUTE;
   }
   pan_merge((global struct mali_job_header_packed *)job, &job_header,
             JOB_HEADER);

   if (is_no_op)
      return;

   global struct mali_compute_payload_packed *payload =
      (global struct mali_compute_payload_packed *)(job +
                                                    pan_section_offset(
                                                       COMPUTE_JOB, PAYLOAD));
   pan_unpack(payload, COMPUTE_PAYLOAD, unpacked);
   pan_pack(payload, COMPUTE_PAYLOAD, cfg) {
      memcpy(&cfg, &unpacked, sizeof(cfg));
      cfg.workgroup_count_x = x;
      cfg.workgroup_count_y = y;
      cfg.workgroup_count_z = 1;
   }
}

/* PATCH v9: geometry shader draw whose vertex and instance counts are only
 * known on the GPU: vkCmdDraw*Indirect with a GS, and a GS after
 * tessellation (the tessellator writes an indexed indirect draw). libpoly
 * fills the vertex/geometry parameters and allocates the buffers from the
 * heap (poly_gs_setup_indirect, as libagx_gs_setup_indirect), then the
 * compute jobs of the draw get their grids, like an indirect dispatch. The
 * rasterization draw reads p->draw as an indirect command. */
KERNEL(1)
panlib_gs_setup_indirect(constant struct panlib_gs_indirect_params *a)
{
   global struct poly_vertex_params *vp =
      (global struct poly_vertex_params *)a->vp;
   global struct poly_geometry_params *p =
      (global struct poly_geometry_params *)a->p;
   constant uint *draw = (constant uint *)a->draw;

   if (a->draw_count &&
       a->draw_index >= *(constant uint *)a->draw_count) {
      p->draw.index_count = 0;
      p->draw.instance_count = 0;
      for (uint i = 0; i < PANLIB_GS_JOBS; i++)
         panlib_gs_patch_job(a->jobs[i], 0, 0);
      return;
   }

   poly_gs_setup_indirect(a->index_buffer, draw, vp, p,
                          (global struct poly_heap *)a->heap, a->vs_outputs,
                          a->index_size_B, a->index_buffer_range_el, a->prim,
                          a->prefix_sum, a->max_indices,
                          (enum poly_gs_shape)a->shape);

   if (a->first_vertex_sysval)
      *(global int *)a->first_vertex_sysval =
         a->index_size_B ? (int)draw[3] : (int)draw[2];
   if (a->base_instance_sysval)
      *(global int *)a->base_instance_sysval =
         a->index_size_B ? (int)draw[4] : (int)draw[3];

   panlib_gs_patch_job(a->jobs[PANLIB_GS_JOB_VS], draw[0], draw[1]);
   panlib_gs_patch_job(a->jobs[PANLIB_GS_JOB_COUNT], p->grid[0], p->grid[1]);
   panlib_gs_patch_job(a->jobs[PANLIB_GS_JOB_MAIN], p->grid[0], p->grid[1]);
   panlib_gs_patch_job(a->jobs[PANLIB_GS_JOB_XFB], p->draw.index_count,
                       p->draw.instance_count);
}

/* PATCH v9: primitive restart before a geometry shader. libpoly assembles
 * the GS input primitives from the index buffer without looking at restart
 * indices, so an indexed strip with restarts gave the GS wrong primitives and
 * primitive IDs (CTS geometry.builtin_variable.in_block.primitive_id_in_restarted).
 * Rewrite the draw as a list without restarts first, like Honeykrisp's
 * hk_draw_without_restart (libagx_unroll_restart), but serial: one invocation
 * walks the index buffer, so no subgroup or local-memory operations are
 * needed. The GS draw then reads out_draw as an indexed indirect draw whose
 * index buffer lives in the heap. */
KERNEL(1)
panlib_unroll_restart(constant struct panlib_unroll_restart_params *a)
{
   global struct poly_heap *heap = (global struct poly_heap *)a->heap;
   constant uint *in_draw = (constant uint *)a->in_draw;
   global uint *out_draw = (global uint *)a->out_draw;
   enum mesa_prim mode = (enum mesa_prim)a->prim;
   const uint index_size_B = a->index_size_B;

   if (a->draw_count &&
       a->draw_index >= *(constant uint *)a->draw_count) {
      out_draw[0] = 0;
      out_draw[1] = 0;
      out_draw[2] = 0;
      out_draw[3] = 0;
      out_draw[4] = 0;
      return;
   }

   const uint count = in_draw[0];
   const uint per_prim = mesa_vertices_per_prim(mode);

   /* Restarts only remove primitives: the list without restarts is the
    * upper bound. */
   const uint max_verts =
      u_decomposed_prims_for_vertices(mode, count) * per_prim;
   const uint alloc_B = poly_heap_alloc_offs(heap, max_verts * index_size_B);
   const uintptr_t out_ptr = (uintptr_t)heap->base + alloc_B;

   out_draw[1] = in_draw[1];                  /* instance count */
   out_draw[2] = alloc_B / index_size_B;      /* first index */
   out_draw[3] = in_draw[3];                  /* vertex offset */
   out_draw[4] = in_draw[4];                  /* first instance */

   const uintptr_t in_ptr = (uintptr_t)poly_index_buffer(
      a->index_buffer, a->index_buffer_range_el, in_draw[2], index_size_B);
   const uint in_range_el =
      poly_index_buffer_range_el(a->index_buffer_range_el, in_draw[2]);

   uint out_prims = 0;
   uint needle = 0;
   while (needle < count) {
      uint next = needle;
      while (next < count && poly_load_index(in_ptr, in_range_el, next,
                                             index_size_B) != a->restart_index)
         next++;

      const uint subprims = u_decomposed_prims_for_vertices(mode, next - needle);
      for (uint i = 0; i < subprims; i++) {
         for (uint vtx = 0; vtx < per_prim; vtx++) {
            uint id = poly_vertex_id_for_topology(mode, a->flatshade_first, i,
                                                  vtx, subprims);
            uint value =
               poly_load_index(in_ptr, in_range_el, needle + id, index_size_B);
            poly_store_index(out_ptr, index_size_B,
                             (out_prims + i) * per_prim + vtx, value);
         }
      }

      out_prims += subprims;
      needle = next + 1;
   }

   out_draw[0] = out_prims * per_prim;
}

static inline void
panlib_stats_inc(uint64_t addr, uint64_t value)
{
   if (addr)
      *(global uint64_t *)addr += value;
}

/* PATCH v9: pipeline statistics queries. Mali has no counters for them, so
 * each draw or dispatch inside a query adds what it does from its API
 * parameters (or its indirect command), and for tessellation and geometry
 * shaders from the draws the tessellator and the GS wrote. Counts that the
 * driver cannot know exactly are upper bounds: primitives of a strip with
 * restarts, and GS output primitives (index count of the GS output, restart
 * separators included). TES invocations come from libpoly
 * (poly_tess_params::statistic) and fragment shader invocations from the
 * occlusion counter. Jobs are serialised with a JM barrier, so plain adds. */
KERNEL(1)
panlib_stats_add(constant struct panlib_stats_params *a)
{
   if (a->draw_count && a->draw_index >= *(constant uint *)a->draw_count)
      return;

   if (a->compute) {
      uint64_t n = a->cs_invocations;
      if (a->cmd) {
         constant uint *d = (constant uint *)a->cmd;
         n = (uint64_t)d[0] * d[1] * d[2] * a->cs_local_size;
      }
      panlib_stats_inc(a->counter[10], n);
      return;
   }

   uint count = a->vertex_count, inst = a->instance_count;
   if (a->cmd) {
      constant uint *d = (constant uint *)a->cmd;
      count = d[0];
      inst = d[1];
   }

   const uint64_t verts = (uint64_t)count * inst;
   const uint64_t prims =
      a->patch_size
         ? (uint64_t)(count / a->patch_size) * inst
         : (uint64_t)u_decomposed_prims_for_vertices((enum mesa_prim)a->prim,
                                                     count) * inst;

   panlib_stats_inc(a->counter[0], verts);
   panlib_stats_inc(a->counter[1], prims);
   panlib_stats_inc(a->counter[2], verts);
   if (a->patch_size)
      panlib_stats_inc(a->counter[8], prims);

   /* Primitives entering the next stage. */
   uint64_t raster = prims;
   if (a->tess_draw) {
      constant uint *d = (constant uint *)a->tess_draw;
      raster = (uint64_t)(d[0] / MAX2(a->tess_verts_per_prim, 1u)) * d[1];
   }

   if (a->gs_invocations) {
      panlib_stats_inc(a->counter[3], raster * a->gs_invocations);
      uint gs_count = a->gs_count, gs_inst = a->gs_instances;
      if (a->gs_draw) {
         constant uint *d = (constant uint *)a->gs_draw;
         gs_count = d[0];
         gs_inst = d[1];
      }
      raster = (uint64_t)gs_count * gs_inst;
      panlib_stats_inc(a->counter[4], raster);
   }

   panlib_stats_inc(a->counter[5], raster);
   panlib_stats_inc(a->counter[6], raster);
}

/* PATCH v9: tessellated vkCmdDraw*Indirect. Same as libagx_tess_setup_indirect
 * (the per-patch buffers come from the heap), then the VS and TCS jobs get
 * their grids. The tessellator kernels loop over p->nr_patches. */
KERNEL(1)
panlib_tess_setup_indirect(constant struct panlib_tess_indirect_params *a)
{
   global struct poly_tess_params *p = (global struct poly_tess_params *)a->tess;
   global struct poly_vertex_params *vp =
      (global struct poly_vertex_params *)a->vp;
   constant uint *draw = (constant uint *)a->draw;

   uint count = draw[0], instance_count = draw[1];
   if (a->draw_count && a->draw_index >= *(constant uint *)a->draw_count)
      count = instance_count = 0;

   uint in_patches = count / p->input_patch_size;
   uint unrolled_patches = in_patches * instance_count;

   uint32_t alloc = 0;
   uint32_t tcs_out_offs = alloc;
   alloc += unrolled_patches * p->tcs_stride_el * 4;
   uint32_t patch_coord_offs = alloc;
   alloc += unrolled_patches * 4;
   uint32_t count_offs = alloc;
   alloc += unrolled_patches * sizeof(uint32_t);
   uint32_t vb_offs = alloc;
   alloc += poly_tcs_in_size(count * instance_count, a->vs_outputs);

   global uchar *blob = poly_heap_alloc(p->heap, alloc);
   p->tcs_buffer = (global float *)(blob + tcs_out_offs);
   p->patches_per_instance = in_patches;
   p->coord_allocs = (global uint *)(blob + patch_coord_offs);
   p->nr_patches = unrolled_patches;
   p->counts = (global uint32_t *)(blob + count_offs);

   vp->output_buffer = (uintptr_t)(blob + vb_offs);
   vp->outputs = a->vs_outputs;
   vp->verts_per_instance = count;

   if (a->index_size_B) {
      vp->index_buffer = poly_index_buffer(a->index_buffer,
                                           a->index_buffer_range_el, draw[2],
                                           a->index_size_B);
      vp->index_buffer_range_el =
         poly_index_buffer_range_el(a->index_buffer_range_el, draw[2]);
   }

   if (a->first_vertex_sysval)
      *(global int *)a->first_vertex_sysval =
         a->index_size_B ? (int)draw[3] : (int)draw[2];
   if (a->base_instance_sysval)
      *(global int *)a->base_instance_sysval =
         a->index_size_B ? (int)draw[4] : (int)draw[3];

   panlib_gs_patch_job(a->jobs[PANLIB_TESS_JOB_VS], count, instance_count);
   panlib_gs_patch_job(a->jobs[PANLIB_TESS_JOB_TCS], in_patches,
                       instance_count);
}

KERNEL(1)
panlib_tes_xfb_setup(global struct panlib_tes_xfb_params *p)
{
   global uint *offsets = (global uint *)p->offsets;
   uint vpp = p->verts_per_prim;
   uint count = *(global uint *)p->draw;

   for (uint b = 0; b < 4; b++) {
      if (!p->stride[b])
         continue;

      uint avail = p->size[b] > offsets[b] ? p->size[b] - offsets[b] : 0;
      count = min(count, avail / (p->stride[b] * vpp) * vpp);
   }

   p->count = count / vpp * vpp;
   panlib_gs_patch_job(p->job, p->count, 1);
}

KERNEL(1)
panlib_tes_xfb_add(global struct panlib_tes_xfb_params *p)
{
   global uint *offsets = (global uint *)p->offsets;

   for (uint b = 0; b < 4; b++)
      offsets[b] += p->count * p->stride[b];
}
#endif
