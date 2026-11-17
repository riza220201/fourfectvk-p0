/*
 * Copyright 2026 NoyszFourFect
 * SPDX-License-Identifier: MIT
 */

#include "compiler/libcl/libcl.h"
#include "xfb_helper.h"

#if PAN_ARCH == 9

/*
 * Transform feedback offset bookkeeping for the v9 Job Manager path.
 *
 * The XFB variant of the vertex shader reads the current byte offset of each
 * XFB buffer from a small array in GPU memory. CSF loads, adds and stores
 * those offsets with command stream instructions. The Job Manager has no such
 * instructions, so these single-invocation kernels are queued in the
 * vertex/tiler/compute chain instead. Bit i of the mask selects buffer i.
 */

KERNEL(1)
panlib_xfb_begin(global uint32_t *offsets, uint32_t mask,
                 global uint32_t *src0, global uint32_t *src1,
                 global uint32_t *src2, global uint32_t *src3)
{
   offsets[0] = (mask & 1) ? *src0 : 0;
   offsets[1] = (mask & 2) ? *src1 : 0;
   offsets[2] = (mask & 4) ? *src2 : 0;
   offsets[3] = (mask & 8) ? *src3 : 0;
}

KERNEL(1)
panlib_xfb_add(global uint32_t *offsets, uint32_t inc0, uint32_t inc1,
               uint32_t inc2, uint32_t inc3)
{
   offsets[0] += inc0;
   offsets[1] += inc1;
   offsets[2] += inc2;
   offsets[3] += inc3;
}

KERNEL(1)
panlib_xfb_end(global uint32_t *offsets, uint32_t mask,
               global uint32_t *dst0, global uint32_t *dst1,
               global uint32_t *dst2, global uint32_t *dst3)
{
   if (mask & 1)
      *dst0 = offsets[0];
   if (mask & 2)
      *dst1 = offsets[1];
   if (mask & 4)
      *dst2 = offsets[2];
   if (mask & 8)
      *dst3 = offsets[3];
}

/* Input vertex of output vertex j, Vulkan spec order for independent
 * primitives (transformFeedbackPreservesProvokingVertex is false). */
static uint32_t
panlib_xfb_map_vertex(uint32_t topology, uint32_t j)
{
   switch (topology) {
   case PANLIB_XFB_LINE_STRIP:
      return j / 2 + j % 2;
   case PANLIB_XFB_TRIANGLE_STRIP: {
      /* Triangle i is i, i + (1 + i % 2), i + (2 - i % 2). */
      uint32_t i = j / 3, k = j % 3;
      return k == 0 ? i : (k == 1 ? i + 1 + (i & 1) : i + 2 - (i & 1));
   }
   case PANLIB_XFB_LINE_LIST_ADJ:
      /* Line i is 4i + 1, 4i + 2. */
      return (j / 2) * 4 + 1 + j % 2;
   case PANLIB_XFB_LINE_STRIP_ADJ:
      /* Line i is i + 1, i + 2. */
      return j / 2 + 1 + j % 2;
   case PANLIB_XFB_TRIANGLE_LIST_ADJ:
      /* Triangle i is 6i, 6i + 2, 6i + 4. */
      return (j / 3) * 6 + (j % 3) * 2;
   case PANLIB_XFB_TRIANGLE_STRIP_ADJ: {
      /* Triangle i is 2i, 2i + 2, 2i + 4 when i is even and
       * 2i + 2, 2i, 2i + 4 when i is odd. */
      uint32_t i = j / 3, k = j % 3;
      if (k == 2 || !(i & 1))
         return 2 * i + 2 * k;
      return 2 * i + (k == 0 ? 2 : 0);
   }
   default: {
      /* Triangle fan: triangle i is i + 1, i + 2, 0. */
      uint32_t i = j / 3, k = j % 3;
      return k == 2 ? 0 : i + 1 + k;
   }
   }
}

KERNEL(64)
panlib_xfb_expand(constant struct panlib_xfb_expand_params *p)
{
   uint32_t id = cl_global_id.x;
   if (id >= p->out_vertices * p->instances)
      return;

   uint32_t inst = id / p->out_vertices;
   uint32_t v = panlib_xfb_map_vertex(p->topology, id % p->out_vertices);
   global uint32_t *offsets = (global uint32_t *)p->offsets;

   for (uint32_t b = 0; b < 4; b++) {
      uint32_t stride = p->stride[b];
      if (!stride)
         continue;

      /* Whole primitives only, nothing past the bound range. */
      uint64_t prim_end =
         (uint64_t)(id / p->verts_per_prim + 1) * p->verts_per_prim * stride;
      if (offsets[b] + prim_end > p->size[b])
         continue;

      global uint32_t *src =
         (global uint32_t *)(p->src[b] +
                             (uint64_t)(inst * p->in_vertices + v) * stride);
      global uint32_t *dst =
         (global uint32_t *)(p->dst[b] + offsets[b] + (uint64_t)id * stride);

      for (uint32_t w = 0; w < stride / 4; w++)
         dst[w] = src[w];
   }
}

#endif
