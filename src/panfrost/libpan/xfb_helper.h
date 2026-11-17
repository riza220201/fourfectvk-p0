/*
 * Copyright 2026 NoyszFourFect
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdint.h>
#include "compiler/libcl/libcl.h"

/* Topologies that need their vertices rewritten into independent primitives
 * before they land in the XFB buffers. Lists and points are captured
 * directly. */
enum panlib_xfb_topology {
   PANLIB_XFB_LINE_STRIP = 0,
   PANLIB_XFB_TRIANGLE_STRIP = 1,
   PANLIB_XFB_TRIANGLE_FAN = 2,
   /* Adjacency without a geometry shader: the adjacent vertices are not
    * captured, only the base lines/triangles. */
   PANLIB_XFB_LINE_LIST_ADJ = 3,
   PANLIB_XFB_LINE_STRIP_ADJ = 4,
   PANLIB_XFB_TRIANGLE_LIST_ADJ = 5,
   PANLIB_XFB_TRIANGLE_STRIP_ADJ = 6,
};

/* Parameters of panlib_xfb_expand(), written by the CPU per draw. */
struct panlib_xfb_expand_params {
   /* Linear per-vertex capture written by the XFB vertex shader variant. */
   uint64_t src[4];
   /* Application XFB buffers. */
   uint64_t dst[4];
   /* u32[4] current byte offset of each XFB buffer. */
   uint64_t offsets;
   /* Bytes per vertex, 0 when the buffer is not written. */
   uint32_t stride[4];
   /* Vertices per instance in the draw and in the captured output. */
   uint32_t in_vertices;
   uint32_t out_vertices;
   uint32_t instances;
   uint32_t topology;
   /* 2 for lines, 3 for triangles. */
   uint32_t verts_per_prim;
   /* Bound size of each XFB buffer in bytes; a primitive that does not fit
    * entirely is not written. */
   uint32_t size[4];
   /* Zero offsets read by the XFB variant while it writes src[]. */
   uint32_t zero[4];
};
