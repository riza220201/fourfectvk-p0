/*
 * Copyright © 2026 Mesa contributors
 * Copyright 2026 NoyszFourFect
 * SPDX-License-Identifier: MIT
 *
 * PATCH v9: transform feedback for the Job Manager path. Same state and
 * shader variant as csf/panvk_vX_cmd_xfb.c. The counter loads and stores
 * that CSF does in the command stream are done by the panlib_xfb_* kernels
 * (libpan/xfb_helper.cl), queued in the current batch.
 */

#include "genxml/gen_macros.h"

#include "panvk_cmd_alloc.h"
#include "panvk_cmd_buffer.h"
#include "panvk_cmd_precomp.h"
#include "panvk_entrypoints.h"

#include "libpan.h"

#if PAN_ARCH == 9

static bool
ensure_xfb_offsets(struct panvk_cmd_buffer *cmdbuf)
{
   if (cmdbuf->state.gfx.xfb.offsets.gpu)
      return true;

   cmdbuf->state.gfx.xfb.offsets = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, PANVK_MAX_XFB_BUFFERS * sizeof(uint32_t),
      sizeof(uint64_t));

   return cmdbuf->state.gfx.xfb.offsets.gpu != 0;
}

static bool
counter_valid(const VkBindTransformFeedbackBuffer2InfoEXT *info)
{
   return info && info->addressRange.address && info->addressRange.size;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBindTransformFeedbackBuffers2EXT)(
   VkCommandBuffer commandBuffer, uint32_t firstBinding,
   uint32_t bindingCount,
   const VkBindTransformFeedbackBuffer2InfoEXT *pBindingInfos)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   assert(firstBinding + bindingCount <= PANVK_MAX_XFB_BUFFERS);

   for (uint32_t i = 0; i < bindingCount; i++) {
      const uint32_t binding = firstBinding + i;

      cmdbuf->state.gfx.xfb.buffers[binding].address =
         pBindingInfos[i].addressRange.address;
      cmdbuf->state.gfx.xfb.buffers[binding].size =
         pBindingInfos[i].addressRange.size;
   }
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBeginTransformFeedback2EXT)(
   VkCommandBuffer commandBuffer, uint32_t firstCounterRange,
   uint32_t counterRangeCount,
   const VkBindTransformFeedbackBuffer2InfoEXT *pCounterInfos)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   assert(!cmdbuf->state.gfx.xfb.enabled);
   assert(firstCounterRange + counterRangeCount <= PANVK_MAX_XFB_BUFFERS);

   if (!ensure_xfb_offsets(cmdbuf))
      return;

   /* Buffer i resumes from its counter when one is given, else from 0. */
   uint64_t src[PANVK_MAX_XFB_BUFFERS] = {0};
   uint32_t mask = 0;
   for (uint32_t i = 0; i < PANVK_MAX_XFB_BUFFERS; i++) {
      const int32_t counter = (int32_t)i - (int32_t)firstCounterRange;

      if (counter >= 0 && counter < (int32_t)counterRangeCount &&
          pCounterInfos && counter_valid(&pCounterInfos[counter])) {
         src[i] = pCounterInfos[counter].addressRange.address;
         mask |= BITFIELD_BIT(i);
      }
   }

   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   panlib_xfb_begin(&precomp_ctx, panlib_1d(1), PANLIB_BARRIER_JM_BARRIER,
                    cmdbuf->state.gfx.xfb.offsets.gpu, mask, src[0], src[1],
                    src[2], src[3]);

   cmdbuf->state.gfx.xfb.enabled = true;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdEndTransformFeedback2EXT)(
   VkCommandBuffer commandBuffer, uint32_t firstCounterRange,
   uint32_t counterRangeCount,
   const VkBindTransformFeedbackBuffer2InfoEXT *pCounterInfos)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   assert(cmdbuf->state.gfx.xfb.enabled);
   assert(firstCounterRange + counterRangeCount <= PANVK_MAX_XFB_BUFFERS);

   uint64_t dst[PANVK_MAX_XFB_BUFFERS] = {0};
   uint32_t mask = 0;
   for (uint32_t i = 0; i < counterRangeCount; i++) {
      if (!pCounterInfos || !counter_valid(&pCounterInfos[i]))
         continue;

      const uint32_t buf = firstCounterRange + i;
      dst[buf] = pCounterInfos[i].addressRange.address;
      mask |= BITFIELD_BIT(buf);
   }

   if (mask) {
      struct panvk_precomp_ctx precomp_ctx =
         panvk_per_arch(precomp_cs)(cmdbuf);
      panlib_xfb_end(&precomp_ctx, panlib_1d(1), PANLIB_BARRIER_JM_BARRIER,
                     cmdbuf->state.gfx.xfb.offsets.gpu, mask, dst[0], dst[1],
                     dst[2], dst[3]);
   }

   cmdbuf->state.gfx.xfb.enabled = false;
}

#endif
