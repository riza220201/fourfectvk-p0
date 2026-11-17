/*
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "compiler/libcl/libcl.h"
#include "compiler/libcl/libcl_vk.h"
#include "poly/geometry.h"
#include "poly/cl/tessellator.h"

#if PAN_ARCH >= 6

/*
 * Mali does not expose fixed-function tessellation.  These kernels are the
 * hardware-independent libpoly tessellator entry points used by PanVK's
 * tessellation-as-compute path.
 */
KERNEL(64)
panlib_tess_isoline(constant struct poly_tess_params *p,
                    enum poly_tess_mode mode__2)
{
#if PAN_ARCH == 9
   /* PATCH v9: grid-stride loop. An indirect draw only knows the patch
    * count on the GPU, so the tessellator is launched with a fixed grid. */
   for (uint i = cl_global_id.x; i < p->nr_patches; i += cl_global_size.x)
      poly_tess_isoline_process(p, i, mode__2);
#else
   if (cl_global_id.x >= p->nr_patches)
      return;
   poly_tess_isoline_process(p, cl_global_id.x, mode__2);
#endif
}

KERNEL(64)
panlib_tess_tri(constant struct poly_tess_params *p,
                enum poly_tess_mode mode__2)
{
#if PAN_ARCH == 9
   /* PATCH v9: grid-stride loop. An indirect draw only knows the patch
    * count on the GPU, so the tessellator is launched with a fixed grid. */
   for (uint i = cl_global_id.x; i < p->nr_patches; i += cl_global_size.x)
      poly_tess_tri_process(p, i, mode__2);
#else
   if (cl_global_id.x >= p->nr_patches)
      return;
   poly_tess_tri_process(p, cl_global_id.x, mode__2);
#endif
}

KERNEL(64)
panlib_tess_quad(constant struct poly_tess_params *p,
                 enum poly_tess_mode mode__2)
{
#if PAN_ARCH == 9
   /* PATCH v9: grid-stride loop. An indirect draw only knows the patch
    * count on the GPU, so the tessellator is launched with a fixed grid. */
   for (uint i = cl_global_id.x; i < p->nr_patches; i += cl_global_size.x)
      poly_tess_quad_process(p, i, mode__2);
#else
   if (cl_global_id.x >= p->nr_patches)
      return;
   poly_tess_quad_process(p, cl_global_id.x, mode__2);
#endif
}

#if PAN_ARCH == 9
/* PATCH v9: the generic version below uses a 1024-thread workgroup and a scan
 * that assumes 32-wide subgroups. Mali-G57 allows 512 threads per workgroup
 * and has 16-wide subgroups, so the job never completes (the vertex/tiler
 * chain times out). One thread does an inclusive scan instead. */
KERNEL(1)
panlib_prefix_sum_tess(global struct poly_tess_params *p)
{
   uint sum = 0;
   for (uint i = 0; i < p->nr_patches; i++) {
      sum += p->counts[i];
      p->counts[i] = sum;
   }

   const uint total = sum;
#else
KERNEL(1024)
panlib_prefix_sum_tess(global struct poly_tess_params *p)
{
   local uint scratch[32];
   poly_prefix_sum(scratch, p->counts, p->nr_patches, 1, 0, 1024);

   barrier(CLK_LOCAL_MEM_FENCE);
   if (cl_local_id.x != 0)
      return;

   const uint total = p->nr_patches ? p->counts[p->nr_patches - 1] : 0;
#endif
   const uint32_t elsize_B = sizeof(uint32_t);
#if PAN_ARCH == 9
   /* PATCH v9: IDVS shades a few indices past the end of the draw. Pad the
    * index buffer with 0xffffffff so those invocations are recognisable
    * (the TES drops their memory writes, panvk_v9_gate_side_effects). */
   const uint pad = 16;
#else
   const uint pad = 0;
#endif
   const uint alloc_B =
      poly_heap_alloc_offs(p->heap, (total + pad) * elsize_B);

   p->index_buffer =
      (global uint32_t *)(((uintptr_t)p->heap->base) + alloc_B);
   for (uint i = 0; i < pad; i++)
      p->index_buffer[total + i] = 0xffffffff;

   global uint32_t *draw = p->out_draws;
   draw[0] = total;
   draw[1] = 1;
   draw[2] = alloc_B / elsize_B;
   draw[3] = 0;
   draw[4] = 0;
}


#endif
