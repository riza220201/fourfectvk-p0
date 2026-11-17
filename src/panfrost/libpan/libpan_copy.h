/*
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */

#pragma once

#define PANLIB_COPY_MEM_WG_SIZE    64
#define PANLIB_COPY_MEM_CHUNK_SIZE 16

/* PANVK_UBO_PUSH (v9): UBO words that draws read from their FAU buffer are
 * copied there by panlib_ubo_gather, at the start of the batch, when the GPU
 * runs it. One entry per 32-bit word. */
#define PANLIB_UBO_GATHER_WG_SIZE 64
#define PANLIB_UBO_GATHER_WG_COUNT 4

/* src is the address of a BUFFER descriptor; offset is applied to the
 * buffer it describes, with a bounds check. Otherwise src is the address of
 * the word itself (0: write 0). */
#define PANLIB_UBO_WORD_FROM_DESC 1

struct panlib_ubo_word {
   uint64_t src;
   uint64_t dst;
   uint32_t offset;
   uint32_t flags;
};

/* Filled when the batch is closed, after the job was queued. */
struct panlib_ubo_gather {
   uint64_t words; /* struct panlib_ubo_word[count] */
   uint32_t count;
   uint32_t pad;
};
