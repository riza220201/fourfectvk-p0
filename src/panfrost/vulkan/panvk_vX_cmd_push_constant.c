/*
 * Copyright © 2024 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "panvk_cmd_alloc.h"
#include "panvk_cmd_buffer.h"
#include "panvk_entrypoints.h"

#if PAN_ARCH == 9
#include "libpan.h"
#include "libpan_copy.h"
#include "panvk_cmd_precomp.h"
#include "panvk_descriptor_set.h"

/* PANVK_UBO_PUSH: where the GPU finds one pushed word. Dynamic buffers are
 * resolved now (their offsets are fixed when the draw is recorded); other
 * UBOs through their descriptor in the set, read by the gather kernel when
 * the batch runs. Unbound or out-of-range words give 0. */
static void
panvk_ubo_push_src(const struct panvk_descriptor_state *desc_state,
                   const struct panvk_ubo_push_word *w,
                   struct panlib_ubo_word *e)
{
   const struct panvk_descriptor_set *set =
      w->set < MAX_SETS ? desc_state->sets[w->set] : NULL;

   e->src = 0;
   e->offset = w->offset;
   e->flags = 0;

   if (!set)
      return;

   if (w->dyn) {
      if (w->idx >= MAX_DYNAMIC_BUFFERS || !set->dyn_bufs[w->idx].dev_addr ||
          (uint64_t)w->offset + 4 > set->dyn_bufs[w->idx].size)
         return;
      e->src = set->dyn_bufs[w->idx].dev_addr +
               desc_state->dyn_buf_offsets[w->set][w->idx] + w->offset;
   } else if (w->idx < set->desc_count && set->descs.dev) {
      e->src = set->descs.dev + (uint64_t)w->idx * PANVK_DESCRIPTOR_SIZE;
      e->flags = PANLIB_UBO_WORD_FROM_DESC;
   }
}

/* The batch's gather job, queued the first time a draw of the batch pushes
 * UBO words. Its entry list is uploaded in cmd_close_batch(). Returns false
 * on allocation failure. */
static bool
panvk_ubo_gather_job(struct panvk_cmd_buffer *cmdbuf)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;

   if (!batch)
      return false;
   if (batch->ubo_gather.hdr)
      return true;

   struct pan_ptr hdr = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, sizeof(struct panlib_ubo_gather), 16);
   if (!hdr.gpu)
      return false;
   memset(hdr.cpu, 0, sizeof(struct panlib_ubo_gather));

   struct panvk_precomp_ctx ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   panlib_ubo_gather(&ctx, panlib_1d(PANLIB_UBO_GATHER_WG_COUNT),
                     PANLIB_BARRIER_NONE, hdr.gpu);
   batch->ubo_gather.hdr = hdr.cpu;
   batch->ubo_gather.job_index = batch->vtc_jc.job_index;
   return true;
}
#endif

static VkResult
prepare_push_uniforms(struct panvk_cmd_buffer *cmdbuf,
                      const struct panvk_shader_variant *shader,
                      struct pan_ptr *push_uniforms, uint32_t repeat_count,
                      uint64_t *sysvals,
                      const struct panvk_descriptor_state *desc_state)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);

   if (!shader->fau.total_count) {
      push_uniforms->cpu = NULL;
      push_uniforms->gpu = 0;
      return VK_SUCCESS;
   }

   *push_uniforms = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, shader->fau.total_count * sizeof(uint64_t) * repeat_count,
      sizeof(uint64_t));

   if (!push_uniforms->gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct panvk_common_sysvals_inner common_inner;
   memcpy(&common_inner, &sysvals[SYSVALS_COMMON_START],
          sizeof(common_inner));
   common_inner.printf_buffer_address = dev->printf.bo->addr.dev;
#if PAN_ARCH < 9
   common_inner.constant_data = panvk_priv_mem_dev_addr(shader->data_mem);
#endif
   uint64_t *common = (uint64_t *)&common_inner;

   uint64_t *push_consts = cmdbuf->state.push_constants.data;
   uint64_t *faus = push_uniforms->cpu;
   const struct pan_fau_layout *fau = &shader->info.fau;
   uint32_t w;

   for (uint32_t r = 0; r < repeat_count; r++) {
      /* Each repeat owns its own contiguous block of total_count slots. */
      uint64_t *repeat_faus = faus + r * shader->fau.total_count;
      uint32_t fau_idx = 0;

      common_inner.push_uniforms =
         push_uniforms->gpu + r * shader->fau.total_count * sizeof(uint64_t);

      /* After packing, the sysvals come first, followed by the user push
       * constants. The ordering is encoded shader side, so don't re-order
       * these loops. */
      BITSET_FOREACH_SET(w, shader->fau.used_sysvals, MAX_SYSVAL_FAUS) {
         if (w >= SYSVALS_COMMON_START && w < SYSVALS_COMMON_END)
            repeat_faus[fau_idx++] = common[w - SYSVALS_COMMON_START];
         else
            repeat_faus[fau_idx++] = sysvals[w];
      }

      BITSET_FOREACH_SET(w, shader->fau.used_push_consts, MAX_PUSH_CONST_FAUS)
         repeat_faus[fau_idx++] = push_consts[w];
      assert(fau_idx <= shader->fau.total_count);

      pan_fau_foreach_imm(fau, i) {
         bool hi = (i & 1) != 0;
         unsigned idx = i / 2;
         assert(fau_idx <= idx && idx < shader->fau.total_count);

         repeat_faus[idx] =
            (repeat_faus[idx] & ((uint64_t)UINT32_MAX << (32 * !hi))) |
            ((uint64_t)fau->words[i].constant << (32 * hi));
      }

#if PAN_ARCH == 9
      if (shader->fau.ubo_push_count && desc_state) {
         struct panvk_batch *batch = cmdbuf->cur_batch;
         uint32_t *words =
            (uint32_t *)&repeat_faus[shader->fau.ubo_push_start];
         const uint64_t words_gpu =
            common_inner.push_uniforms +
            shader->fau.ubo_push_start * sizeof(uint64_t);

         assert(shader->fau.ubo_push_start +
                   DIV_ROUND_UP(shader->fau.ubo_push_count, 2) <=
                shader->fau.total_count);

         if (!panvk_ubo_gather_job(cmdbuf))
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;

         for (uint32_t i = 0; i < shader->fau.ubo_push_count; i++) {
            struct panlib_ubo_word e;

            panvk_ubo_push_src(desc_state, &shader->fau.ubo_push[i], &e);
            e.dst = words_gpu + i * sizeof(uint32_t);
            words[i] = 0;
            util_dynarray_append(&batch->ubo_gather.words, e);
         }
      }
#endif
   }

   return VK_SUCCESS;
}

VkResult
panvk_per_arch(cmd_prepare_gfx_push_uniforms)(
   struct panvk_cmd_buffer *cmdbuf, const struct panvk_shader_variant *shader,
   struct pan_ptr *push_uniforms, uint32_t repeat_count)
{
   return prepare_push_uniforms(cmdbuf, shader, push_uniforms, repeat_count,
                                (uint64_t *)&cmdbuf->state.gfx.sysvals,
                                &cmdbuf->state.gfx.desc_state);
}

VkResult
panvk_per_arch(cmd_prepare_compute_push_uniforms)(
   struct panvk_cmd_buffer *cmdbuf, const struct panvk_shader_variant *shader,
   struct pan_ptr *push_uniforms)
{
   return prepare_push_uniforms(cmdbuf, shader, push_uniforms, 1,
                                (uint64_t *)&cmdbuf->state.compute.sysvals,
                                &cmdbuf->state.compute.desc_state);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdPushConstants2KHR)(
   VkCommandBuffer commandBuffer,
   const VkPushConstantsInfoKHR *pPushConstantsInfo)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (pPushConstantsInfo->stageFlags &
       (VK_SHADER_STAGE_VERTEX_BIT |
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT))
      gfx_state_set_dirty(cmdbuf, VS_PUSH_UNIFORMS);

   if (pPushConstantsInfo->stageFlags & VK_SHADER_STAGE_FRAGMENT_BIT)
      gfx_state_set_dirty(cmdbuf, FS_PUSH_UNIFORMS);

   if (pPushConstantsInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
      compute_state_set_dirty(cmdbuf, PUSH_UNIFORMS);

   uint8_t *data =
      (uint8_t *)cmdbuf->state.push_constants.data + pPushConstantsInfo->offset;

   memcpy(data, pPushConstantsInfo->pValues, pPushConstantsInfo->size);
}
