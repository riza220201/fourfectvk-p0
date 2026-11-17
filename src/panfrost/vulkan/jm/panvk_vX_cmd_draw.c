/*
 * Copyright © 2024 Collabora Ltd.
 * Copyright © 2026 NXP
 *
 * Derived from tu_cmd_buffer.c which is:
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "genxml/gen_macros.h"
#include "pan_v9_trace.h"

#include "panvk_buffer.h"
#include "panvk_cmd_alloc.h"
#include "panvk_cmd_buffer.h"
#include "panvk_cmd_desc_state.h"
#include "panvk_cmd_draw.h"
#include "panvk_cmd_meta.h"
#include "panvk_cmd_precomp.h"
#include "xfb_helper.h" /* PATCH v9: JM transform feedback */
#include "geometry_helper.h" /* PATCH v9: JM indirect geometry shaders */
#include "poly/geometry.h"   /* PATCH v9: JM tessellation */
#include "poly/tessellator.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_image.h"
#include "panvk_image_view.h"

/* PATCH: seluruh jalur draw grafis (vertex/tiler/fragment) di file ini masih
 * ditulis untuk Bifrost (RENDERER_STATE/rsd, dst) dan BELUM diadaptasi untuk
 * v9/Valhall -- itu pekerjaan porting terpisah yang jauh lebih besar dari
 * compute. Untuk PAN_ARCH >= 9, kode asli di-skip dan diganti stub kosong
 * (lihat #else di akhir file) supaya libvulkan_panfrost.so bisa selesai
 * di-link dan compute (yang sudah diadaptasi di jm_panvk_vX_cmd_dispatch.c)
 * bisa divalidasi malam ini. Draw call APA PUN lewat jalur ini akan
 * no-op/gagal untuk v9 sampai porting grafisnya dikerjakan. */
#include "panvk_instance.h"
#include "panvk_meta.h"
#include "panvk_priv_bo.h"
#include "panvk_shader.h"

#include "draw_helper.h"
#include "pan_desc.h"
#include "pan_earlyzs.h"
#include "pan_encoder.h"
#include "pan_format.h"
#include "pan_jc.h"
#include "pan_props.h"
#include "pan_shader.h"

#include "vk_format.h"
#include "vk_meta.h"
#include "vk_pipeline_layout.h"
#include "panfrost/lib/pan_env.h"

struct panvk_draw_data {
   struct panvk_draw_info info;
   unsigned vertex_range;
   unsigned padded_vertex_count;
#if PAN_ARCH < 9
   struct mali_invocation_packed invocation;
#endif
   struct {
      uint64_t varyings;
      uint64_t attributes;
      uint64_t attribute_bufs;
   } vs;
   struct {
      uint64_t rsd;
      uint64_t varyings;
   } fs;
   uint64_t varying_bufs;
   uint64_t position;
   union {
      uint64_t psiz;
      float line_width;
   };
   uint64_t tls;
   uint64_t fb;
   const struct pan_tiler_context *tiler_ctx;
   uint64_t viewport;
   /* PATCH v9: blend descriptor dialokasikan & diisi LEBIH AWAL (sebelum
    * cmd_prepare_draw_sysvals/push-uniforms) - root cause "clear-only"
    * (ditemukan komunitas): FAU sempat kebawa nilai blend_descs[] basi
    * krn ke-baca sysval sebelum blend_emit_descs() sempat jalan. */
   struct pan_ptr blend_ptr;
   uint32_t bd_count;
   struct {
      struct pan_ptr vertex_copy_desc;
      struct pan_ptr frag_copy_desc;
      union {
         struct {
            struct pan_ptr vertex;
            struct pan_ptr tiler;
         };
         struct pan_ptr idvs;
      };
   } jobs;
   struct {
      uint64_t attribs;
      uint64_t attrib_bufs;
      uint64_t varying_bufs;
   } indirect_info;
};

static bool
is_indirect_draw(const struct panvk_draw_data *draw)
{
   return draw->info.indirect.buffer_dev_addr != 0 ||
          draw->info.index.index_size != 0;
}

#if PAN_ARCH < 9 /* PATCH: guard digeser ke sini, struct+includes di atas jadi unconditional */

static bool
has_depth_att(struct panvk_cmd_buffer *cmdbuf)
{
   return (cmdbuf->state.gfx.render.bound_attachments &
           MESA_VK_RP_ATTACHMENT_DEPTH_BIT) != 0;
}

static bool
has_stencil_att(struct panvk_cmd_buffer *cmdbuf)
{
   return (cmdbuf->state.gfx.render.bound_attachments &
           MESA_VK_RP_ATTACHMENT_STENCIL_BIT) != 0;
}

static bool
writes_depth(struct panvk_cmd_buffer *cmdbuf)
{
   const struct vk_depth_stencil_state *ds =
      &cmdbuf->vk.dynamic_graphics_state.ds;

   return has_depth_att(cmdbuf) && ds->depth.test_enable &&
          ds->depth.write_enable && ds->depth.compare_op != VK_COMPARE_OP_NEVER;
}

static bool
writes_stencil(struct panvk_cmd_buffer *cmdbuf)
{
   const struct vk_depth_stencil_state *ds =
      &cmdbuf->vk.dynamic_graphics_state.ds;

   return has_stencil_att(cmdbuf) && ds->stencil.test_enable &&
          ((ds->stencil.front.write_mask &&
            (ds->stencil.front.op.fail != VK_STENCIL_OP_KEEP ||
             ds->stencil.front.op.pass != VK_STENCIL_OP_KEEP ||
             ds->stencil.front.op.depth_fail != VK_STENCIL_OP_KEEP)) ||
           (ds->stencil.back.write_mask &&
            (ds->stencil.back.op.fail != VK_STENCIL_OP_KEEP ||
             ds->stencil.back.op.pass != VK_STENCIL_OP_KEEP ||
             ds->stencil.back.op.depth_fail != VK_STENCIL_OP_KEEP)));
}

static bool
ds_test_always_passes(struct panvk_cmd_buffer *cmdbuf)
{
   const struct vk_depth_stencil_state *ds =
      &cmdbuf->vk.dynamic_graphics_state.ds;

   if (!has_depth_att(cmdbuf))
      return true;

   if (ds->depth.test_enable && ds->depth.compare_op != VK_COMPARE_OP_ALWAYS)
      return false;

   if (ds->stencil.test_enable &&
       (ds->stencil.front.op.compare != VK_COMPARE_OP_ALWAYS ||
        ds->stencil.back.op.compare != VK_COMPARE_OP_ALWAYS))
      return false;

   return true;
}

static inline enum mali_func
translate_compare_func(VkCompareOp comp)
{
   STATIC_ASSERT(VK_COMPARE_OP_NEVER == (VkCompareOp)MALI_FUNC_NEVER);
   STATIC_ASSERT(VK_COMPARE_OP_LESS == (VkCompareOp)MALI_FUNC_LESS);
   STATIC_ASSERT(VK_COMPARE_OP_EQUAL == (VkCompareOp)MALI_FUNC_EQUAL);
   STATIC_ASSERT(VK_COMPARE_OP_LESS_OR_EQUAL == (VkCompareOp)MALI_FUNC_LEQUAL);
   STATIC_ASSERT(VK_COMPARE_OP_GREATER == (VkCompareOp)MALI_FUNC_GREATER);
   STATIC_ASSERT(VK_COMPARE_OP_NOT_EQUAL == (VkCompareOp)MALI_FUNC_NOT_EQUAL);
   STATIC_ASSERT(VK_COMPARE_OP_GREATER_OR_EQUAL ==
                 (VkCompareOp)MALI_FUNC_GEQUAL);
   STATIC_ASSERT(VK_COMPARE_OP_ALWAYS == (VkCompareOp)MALI_FUNC_ALWAYS);

   return (enum mali_func)comp;
}

static enum mali_stencil_op
translate_stencil_op(VkStencilOp in)
{
   switch (in) {
   case VK_STENCIL_OP_KEEP:
      return MALI_STENCIL_OP_KEEP;
   case VK_STENCIL_OP_ZERO:
      return MALI_STENCIL_OP_ZERO;
   case VK_STENCIL_OP_REPLACE:
      return MALI_STENCIL_OP_REPLACE;
   case VK_STENCIL_OP_INCREMENT_AND_CLAMP:
      return MALI_STENCIL_OP_INCR_SAT;
   case VK_STENCIL_OP_DECREMENT_AND_CLAMP:
      return MALI_STENCIL_OP_DECR_SAT;
   case VK_STENCIL_OP_INCREMENT_AND_WRAP:
      return MALI_STENCIL_OP_INCR_WRAP;
   case VK_STENCIL_OP_DECREMENT_AND_WRAP:
      return MALI_STENCIL_OP_DECR_WRAP;
   case VK_STENCIL_OP_INVERT:
      return MALI_STENCIL_OP_INVERT;
   default:
      UNREACHABLE("Invalid stencil op");
   }
}

static VkResult
panvk_draw_prepare_fs_rsd(struct panvk_cmd_buffer *cmdbuf,
                          struct panvk_draw_data *draw)
{
   bool dirty = dyn_gfx_state_dirty(cmdbuf, RS_RASTERIZER_DISCARD_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_CLAMP_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_CLIP_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_BIAS_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_BIAS_FACTORS) ||
                dyn_gfx_state_dirty(cmdbuf, RS_LINE_MODE) ||
                /* line mode needs primitive topology */
                dyn_gfx_state_dirty(cmdbuf, IA_PRIMITIVE_TOPOLOGY) ||
                dyn_gfx_state_dirty(cmdbuf, CB_LOGIC_OP_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, CB_LOGIC_OP) ||
                dyn_gfx_state_dirty(cmdbuf, CB_ATTACHMENT_COUNT) ||
                dyn_gfx_state_dirty(cmdbuf, CB_COLOR_WRITE_ENABLES) ||
                dyn_gfx_state_dirty(cmdbuf, CB_BLEND_ENABLES) ||
                dyn_gfx_state_dirty(cmdbuf, CB_BLEND_EQUATIONS) ||
                dyn_gfx_state_dirty(cmdbuf, CB_WRITE_MASKS) ||
                dyn_gfx_state_dirty(cmdbuf, CB_BLEND_CONSTANTS) ||
                dyn_gfx_state_dirty(cmdbuf, COLOR_ATTACHMENT_MAP) ||
                dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_TEST_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_WRITE_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_COMPARE_OP) ||
                dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_COMPARE_OP) ||
                dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_TEST_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_OP) ||
                dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_COMPARE_MASK) ||
                dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_WRITE_MASK) ||
                dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_REFERENCE) ||
                dyn_gfx_state_dirty(cmdbuf, MS_RASTERIZATION_SAMPLES) ||
                dyn_gfx_state_dirty(cmdbuf, MS_SAMPLE_MASK) ||
                dyn_gfx_state_dirty(cmdbuf, MS_ALPHA_TO_COVERAGE_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, MS_ALPHA_TO_ONE_ENABLE) ||
                gfx_state_dirty(cmdbuf, FS) || gfx_state_dirty(cmdbuf, OQ) ||
                gfx_state_dirty(cmdbuf, RENDER_STATE);

   if (!dirty) {
      draw->fs.rsd = cmdbuf->state.gfx.fs.rsd;
      return VK_SUCCESS;
   }

   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_rasterization_state *rs = &dyns->rs;
   const struct vk_depth_stencil_state *ds = &dyns->ds;
   const struct vk_input_assembly_state *ia = &dyns->ia;
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   const struct pan_shader_info *fs_info = fs ? &fs->info : NULL;
   uint32_t bd_count = cmdbuf->state.gfx.render.fb.layout.rt_count;
   bool test_s = has_stencil_att(cmdbuf) && ds->stencil.test_enable;
   bool test_z = has_depth_att(cmdbuf) && ds->depth.test_enable;
   bool writes_z = writes_depth(cmdbuf);
   bool writes_s = writes_stencil(cmdbuf);

   bool msaa = dyns->ms.rasterization_samples > 1;
   if ((ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
        ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP) &&
       rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM) {
      /* we need to disable MSAA when rendering bresenham lines.
       *
       * From the Vulkan spec:
       *   "When Bresenham lines are being rasterized, sample locations may
       *    all be treated as being at the pixel center (this may affect
       *    attribute and depth interpolation).""
       */
      msaa = false;
   }

   struct pan_ptr ptr = panvk_cmd_alloc_desc_aggregate(
      cmdbuf, PAN_DESC(RENDERER_STATE), PAN_DESC_ARRAY(bd_count, BLEND));
   if (!ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct mali_renderer_state_packed *rsd = ptr.cpu;
   struct mali_blend_packed *bds = ptr.cpu + pan_size(RENDERER_STATE);
   struct panvk_blend_info *binfo = &cmdbuf->state.gfx.cb.info;

   uint64_t fs_code = panvk_shader_variant_get_dev_addr(fs);

   if (fs_info != NULL) {
      panvk_per_arch(blend_emit_descs)(cmdbuf, bds);
   } else {
      for (unsigned i = 0; i < bd_count; i++) {
         pan_pack(&bds[i], BLEND, cfg) {
            cfg.enable = false;
            cfg.internal.mode = MALI_BLEND_MODE_OFF;
         }
      }
   }

   pan_pack(rsd, RENDERER_STATE, cfg) {
      bool alpha_to_coverage = dyns->ms.alpha_to_coverage_enable;

      if (fs) {
         pan_shader_prepare_rsd(fs_info, fs_code, &cfg);

         uint8_t rt_mask = cmdbuf->state.gfx.render.bound_attachments &
                           MESA_VK_RP_ATTACHMENT_ANY_COLOR_BITS;
         uint8_t rt_written = color_attachment_written_mask(
            fs, &cmdbuf->vk.dynamic_graphics_state.cal);
         uint8_t rt_read = color_attachment_read_mask(fs, &dyns->ial, rt_mask);
         enum pan_earlyzs_zs_tilebuf_read zs_read =
            (z_attachment_read(fs, &dyns->ial) ||
             s_attachment_read(fs, &dyns->ial))
               ? PAN_EARLYZS_ZS_TILEBUF_READ_NO_OPT
               : PAN_EARLYZS_ZS_TILEBUF_NOT_READ;

         cfg.properties.allow_forward_pixel_to_kill =
            fs_info->fs.can_fpk && !(rt_mask & ~rt_written) &&
            !(rt_read & rt_written) && !alpha_to_coverage &&
            !binfo->any_dest_read;

         bool writes_zs = writes_z || writes_s;
         bool zs_always_passes = ds_test_always_passes(cmdbuf);
         bool oq = cmdbuf->state.gfx.occlusion_query.mode !=
                   MALI_OCCLUSION_MODE_DISABLED;

         struct pan_earlyzs_state earlyzs =
            pan_earlyzs_get(fs->fs.earlyzs_lut, writes_zs || oq,
                            alpha_to_coverage, zs_always_passes, zs_read);

         /* early ZS check for FPK is performed by HW on v7+ */
         cfg.properties.allow_forward_pixel_to_be_killed =
            !fs->info.writes_global &&
            ((PAN_ARCH > 6) || earlyzs.kill != MALI_PIXEL_KILL_FORCE_LATE);

         cfg.properties.pixel_kill_operation = earlyzs.kill;
         cfg.properties.zs_update_operation = earlyzs.update;
         cfg.multisample_misc.evaluate_per_sample =
            (fs->info.fs.sample_shading && dyns->ms.rasterization_samples > 1);
      } else {
         cfg.properties.depth_source = MALI_DEPTH_SOURCE_FIXED_FUNCTION;
         cfg.properties.allow_forward_pixel_to_kill = true;
         cfg.properties.allow_forward_pixel_to_be_killed = true;
         cfg.properties.zs_update_operation = MALI_PIXEL_KILL_FORCE_EARLY;
      }

      cfg.multisample_misc.multisample_enable = msaa;
      cfg.multisample_misc.sample_mask = dyns->ms.sample_mask;

      cfg.multisample_misc.depth_function =
         test_z ? translate_compare_func(ds->depth.compare_op)
                : MALI_FUNC_ALWAYS;

      cfg.multisample_misc.depth_write_mask = writes_z;
      cfg.multisample_misc.fixed_function_near_discard =
      cfg.multisample_misc.fixed_function_far_discard =
         vk_rasterization_state_depth_clip_enable(rs);
      cfg.multisample_misc.fixed_function_depth_range_fixed =
         !rs->depth_clamp_enable;
      cfg.multisample_misc.shader_depth_range_fixed = true;

      cfg.stencil_mask_misc.stencil_enable = test_s;
      cfg.stencil_mask_misc.alpha_to_coverage = alpha_to_coverage;
      cfg.stencil_mask_misc.alpha_test_compare_function = MALI_FUNC_ALWAYS;
      cfg.stencil_mask_misc.front_facing_depth_bias = rs->depth_bias.enable;
      cfg.stencil_mask_misc.back_facing_depth_bias = rs->depth_bias.enable;

      if (rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM)
         cfg.stencil_mask_misc.aligned_line_ends = true;

      cfg.depth_units = rs->depth_bias.constant_factor;
      cfg.depth_factor = rs->depth_bias.slope_factor;
      cfg.depth_bias_clamp = rs->depth_bias.clamp;

      cfg.stencil_front.mask = ds->stencil.front.compare_mask;
      cfg.stencil_back.mask = ds->stencil.back.compare_mask;

      cfg.stencil_mask_misc.stencil_mask_front = ds->stencil.front.write_mask;
      cfg.stencil_mask_misc.stencil_mask_back = ds->stencil.back.write_mask;

      cfg.stencil_front.reference_value = ds->stencil.front.reference;
      cfg.stencil_back.reference_value = ds->stencil.back.reference;

      if (test_s) {
         cfg.stencil_front.compare_function =
            translate_compare_func(ds->stencil.front.op.compare);
         cfg.stencil_front.stencil_fail =
            translate_stencil_op(ds->stencil.front.op.fail);
         cfg.stencil_front.depth_fail =
            translate_stencil_op(ds->stencil.front.op.depth_fail);
         cfg.stencil_front.depth_pass =
            translate_stencil_op(ds->stencil.front.op.pass);
         cfg.stencil_back.compare_function =
            translate_compare_func(ds->stencil.back.op.compare);
         cfg.stencil_back.stencil_fail =
            translate_stencil_op(ds->stencil.back.op.fail);
         cfg.stencil_back.depth_fail =
            translate_stencil_op(ds->stencil.back.op.depth_fail);
         cfg.stencil_back.depth_pass =
            translate_stencil_op(ds->stencil.back.op.pass);
      }
   }

   cmdbuf->state.gfx.fs.rsd = ptr.gpu;
   draw->fs.rsd = cmdbuf->state.gfx.fs.rsd;
   return VK_SUCCESS;
}

static VkResult
panvk_draw_prepare_tiler_context(struct panvk_cmd_buffer *cmdbuf,
                                 struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   VkResult result =
      panvk_per_arch(cmd_prepare_tiler_context)(cmdbuf, draw->info.layer_id);
   if (result != VK_SUCCESS)
      return result;

   draw->tiler_ctx = &batch->tiler.ctx;
   return VK_SUCCESS;
}

static VkResult
panvk_draw_prepare_varyings(struct panvk_cmd_buffer *cmdbuf,
                            struct panvk_draw_data *draw)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_link *link = &cmdbuf->state.gfx.link;
   struct pan_ptr bufs = panvk_cmd_alloc_desc_array(
      cmdbuf, PANVK_VARY_BUF_MAX + 1, ATTRIBUTE_BUFFER);
   if (!bufs.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct mali_attribute_buffer_packed *buf_descs = bufs.cpu;
   const struct vk_input_assembly_state *ia =
      &cmdbuf->vk.dynamic_graphics_state.ia;
   bool writes_point_size =
      vs->info.vs.writes_point_size &&
      ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   uint64_t psiz_buf = 0;

   if (is_indirect_draw(draw) &&
       !cmdbuf->state.gfx.vs.indirect_varying_bufs_infos) {
      struct pan_ptr bufs_info_storage = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc, sizeof(struct libpan_draw_helper_varying_buf_info), 8);

      if (!bufs_info_storage.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      cmdbuf->state.gfx.vs.indirect_varying_bufs_infos = bufs_info_storage.gpu;

      struct libpan_draw_helper_varying_buf_info *vary_bufs_info =
         bufs_info_storage.cpu;
      vary_bufs_info->address = dev->indirect_varying_buffer->addr.dev;
      vary_bufs_info->size = PANVK_JM_MAX_PER_VTX_ATTRIBUTES_INDIRECT_SIZE *
                             PANVK_JM_MAX_VERTICES_INDIRECT;
      vary_bufs_info->offset = 0;
   }

   for (unsigned i = 0; i < PANVK_VARY_BUF_MAX; i++) {
      uint32_t buf_size;
      uint64_t buf_addr;
      if (is_indirect_draw(draw)) {
         buf_addr = dev->indirect_varying_buffer->addr.dev;
         buf_size = 0;
      } else {
         buf_size = draw->padded_vertex_count * draw->info.instance.count *
                    link->buf_strides[i];
         buf_addr =
            buf_size
               ? panvk_cmd_alloc_dev_mem(cmdbuf, varying, buf_size, 64).gpu
               : 0;

         if (buf_size && !buf_addr)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      }

      pan_pack(&buf_descs[i], ATTRIBUTE_BUFFER, cfg) {
         cfg.stride = link->buf_strides[i];
         cfg.size = buf_size;
         cfg.pointer = buf_addr;
      }

      if (i == PANVK_VARY_BUF_POSITION)
         draw->position = buf_addr;

      if (i == PANVK_VARY_BUF_PSIZ)
         psiz_buf = buf_addr;
   }

   /* We need an empty entry to stop prefetching on Bifrost */
   memset(bufs.cpu + (pan_size(ATTRIBUTE_BUFFER) * PANVK_VARY_BUF_MAX), 0,
          pan_size(ATTRIBUTE_BUFFER));

   if (writes_point_size)
      draw->psiz = psiz_buf;
   else if (ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
            ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP)
      draw->line_width = cmdbuf->vk.dynamic_graphics_state.rs.line.width;
   else
      draw->line_width = 1.0f;

   draw->varying_bufs = bufs.gpu;
   draw->indirect_info.varying_bufs =
      cmdbuf->state.gfx.vs.indirect_varying_bufs_infos;
   draw->vs.varyings = panvk_priv_mem_dev_addr(link->vs.attribs);
   draw->fs.varyings = panvk_priv_mem_dev_addr(link->fs.attribs);
   return VK_SUCCESS;
}

static void
panvk_draw_emit_attrib_buf(
   const struct panvk_draw_data *draw,
   const struct vk_vertex_binding_state *buf_info, uint32_t stride,
   const struct panvk_attrib_buf *buf,
   struct mali_attribute_buffer_packed *desc,
   struct libpan_draw_helper_attrib_buf_info *helper_buf_info)
{
   uint64_t addr = buf->address & ~63ULL;
   unsigned size = buf->size + (buf->address & 63);
   unsigned divisor = draw->padded_vertex_count * buf_info->divisor;
   bool per_instance = buf_info->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
   struct mali_attribute_buffer_packed *buf_ext = &desc[1];

   /* In case of indirect draw, the descriptor will be patched at runtime */
   if (helper_buf_info != NULL) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.pointer = addr;
         cfg.size = size;
      }

      helper_buf_info->divisor = buf_info->divisor;
      helper_buf_info->stride = stride;
      helper_buf_info->per_instance = per_instance;
   } else if (draw->info.instance.count <= 1) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.stride = per_instance ? 0 : stride;
         cfg.pointer = addr;
         cfg.size = size;
      }
   } else if (!per_instance) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_MODULUS;
         cfg.divisor = draw->padded_vertex_count;
         cfg.stride = stride;
         cfg.pointer = addr;
         cfg.size = size;
      }
   } else if (!divisor) {
      /* instance_divisor == 0 means all instances share the same value.
       * Make it a 1D array with a zero stride.
       */
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.stride = 0;
         cfg.pointer = addr;
         cfg.size = size;
      }
   } else if (util_is_power_of_two_or_zero(divisor)) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_POT_DIVISOR;
         cfg.stride = stride;
         cfg.pointer = addr;
         cfg.size = size;
         cfg.divisor_r = __builtin_ctz(divisor);
      }
   } else {
      unsigned divisor_r = 0, divisor_e = 0;
      unsigned divisor_d =
         pan_compute_npot_divisor(divisor, &divisor_r, &divisor_e);
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_NPOT_DIVISOR;
         cfg.stride = stride;
         cfg.pointer = addr;
         cfg.size = size;
         cfg.divisor_r = divisor_r;
         cfg.divisor_e = divisor_e;
      }

      pan_cast_and_pack(buf_ext, ATTRIBUTE_BUFFER_CONTINUATION_NPOT, cfg) {
         cfg.divisor_numerator = divisor_d;
         cfg.divisor = buf_info->divisor;
      }

      buf_ext = NULL;
   }

   /* If the buffer extension wasn't used, memset(0) */
   if (buf_ext)
      memset(buf_ext, 0, pan_size(ATTRIBUTE_BUFFER));
}

static void
panvk_draw_emit_attrib(const struct panvk_draw_data *draw,
                       const struct vk_vertex_attribute_state *attrib_info,
                       const struct vk_vertex_binding_state *buf_info,
                       const struct panvk_attrib_buf *buf,
                       struct mali_attribute_packed *desc,
                       struct libpan_draw_helper_attrib_info *helper_attrib_info)
{
   bool per_instance = buf_info->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
   enum pipe_format f = vk_format_to_pipe_format(attrib_info->format);
   unsigned buf_idx = attrib_info->binding;

   pan_pack(desc, ATTRIBUTE, cfg) {
      cfg.buffer_index = buf_idx * 2;
      cfg.offset_enable = true;
      cfg.format = GENX(pan_format_from_pipe_format)(f)->hw;

      uint32_t offset = attrib_info->offset + (buf->address & 63);

      /* In case of indirect draw, the descriptor will be patched at runtime */
      if (helper_attrib_info != NULL) {
         helper_attrib_info->base_offset = offset;
         helper_attrib_info->stride = per_instance ? buf_info->stride : 0;
      } else {
         cfg.offset = offset;
         if (per_instance)
            cfg.offset += draw->info.instance.base * buf_info->stride;
      }
   }
}

/* Don't re-use vertex attributes, always re-emit them.  Descriptors depend on
 * per-draw parameters (e.g. vertex_count, instancing), which are likely to
 * change every draw, so don't bother trying to save an attribute[_buffer]
 * re-emission.
 */
static VkResult
panvk_draw_prepare_vs_attribs(struct panvk_cmd_buffer *cmdbuf,
                              struct panvk_draw_data *draw)
{
   const struct panvk_shader *vs = cmdbuf->state.gfx.vs.shader;
   const struct panvk_shader_variant *vs_variant =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_vertex_input_state *vi = dyns->vi;
   unsigned num_imgs = vs->desc_info.others.count[PANVK_BIFROST_DESC_TABLE_IMG];
   unsigned num_vbs = util_last_bit(vi->bindings_valid);
   unsigned num_vs_attribs =
      num_imgs ? MAX_VS_ATTRIBS
               : MAX2(util_last_bit(vi->attributes_valid),
                      vs_variant->info.attribute_count);
   unsigned attrib_count = num_vs_attribs + num_imgs;

   unsigned attrib_buf_count = (num_vbs + num_imgs) * 2;
   struct pan_ptr bufs = panvk_cmd_alloc_desc_array(
      cmdbuf, attrib_buf_count + 1, ATTRIBUTE_BUFFER);
   struct mali_attribute_buffer_packed *attrib_buf_descs = bufs.cpu;
   struct pan_ptr attribs =
      panvk_cmd_alloc_desc_array(cmdbuf, attrib_count, ATTRIBUTE);
   struct mali_attribute_packed *attrib_descs = attribs.cpu;

   if (!bufs.gpu || (attrib_count && !attribs.gpu))
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct libpan_draw_helper_attrib_buf_info *bufs_infos = NULL;
   struct libpan_draw_helper_attrib_info *attribs_infos = NULL;

   if (is_indirect_draw(draw)) {
      struct pan_ptr bufs_infos_storage = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc,
         num_vbs * sizeof(struct libpan_draw_helper_attrib_buf_info), 8);
      struct pan_ptr attribs_infos_storage = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc,
         num_vs_attribs * sizeof(struct libpan_draw_helper_attrib_info), 8);

      if (!bufs_infos_storage.gpu ||
          (num_vs_attribs && !attribs_infos_storage.gpu))
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      cmdbuf->state.gfx.vs.indirect_attrib_bufs_infos = bufs_infos_storage.gpu;
      cmdbuf->state.gfx.vs.indirect_attribs_infos = attribs_infos_storage.gpu;
      bufs_infos = bufs_infos_storage.cpu;
      attribs_infos = attribs_infos_storage.cpu;
   }

   for (unsigned i = 0; i < num_vbs; i++) {
      if (vi->bindings_valid & BITFIELD_BIT(i)) {
         struct libpan_draw_helper_attrib_buf_info *helper_buf_info =
            bufs_infos ? &bufs_infos[i] : NULL;
         panvk_draw_emit_attrib_buf(draw, &vi->bindings[i],
                                    dyns->vi_binding_strides[i],
                                    &cmdbuf->state.gfx.vb.bufs[i],
                                    &attrib_buf_descs[i * 2], helper_buf_info);
      } else {
         memset(&attrib_buf_descs[i * 2], 0, sizeof(*attrib_buf_descs) * 2);
      }
   }

   for (unsigned i = 0; i < num_vs_attribs; i++) {
      if (vi->attributes_valid & BITFIELD_BIT(i)) {
         unsigned buf_idx = vi->attributes[i].binding;
         struct libpan_draw_helper_attrib_info *helper_attrib_info =
            attribs_infos ? &attribs_infos[i] : NULL;
         panvk_draw_emit_attrib(draw, &vi->attributes[i],
                                &vi->bindings[buf_idx],
                                &cmdbuf->state.gfx.vb.bufs[buf_idx],
                                &attrib_descs[i], helper_attrib_info);
      } else {
         /* Unbound attributes read the zero default */
         pan_pack(&attrib_descs[i], ATTRIBUTE, cfg)
            cfg.format = MALI_PACK_FMT(CONSTANT, 0000, L);
      }
   }

   /* A NULL entry is needed to stop prefecting on Bifrost */
   memset(bufs.cpu + (pan_size(ATTRIBUTE_BUFFER) * attrib_buf_count), 0,
          pan_size(ATTRIBUTE_BUFFER));

   cmdbuf->state.gfx.vs.attrib_bufs = bufs.gpu;
   cmdbuf->state.gfx.vs.attribs = attribs.gpu;

   if (num_imgs) {
      cmdbuf->state.gfx.vs.desc.img_attrib_table =
         attribs.gpu + (MAX_VS_ATTRIBS * pan_size(ATTRIBUTE));
      cmdbuf->state.gfx.vs.desc.tables[PANVK_BIFROST_DESC_TABLE_IMG] =
         bufs.gpu + (num_vbs * pan_size(ATTRIBUTE_BUFFER) * 2);
   }

   return VK_SUCCESS;
}

static void
panvk_draw_prepare_attributes(struct panvk_cmd_buffer *cmdbuf,
                              struct panvk_draw_data *draw)
{
   panvk_draw_prepare_vs_attribs(cmdbuf, draw);
   draw->vs.attributes = cmdbuf->state.gfx.vs.attribs;
   draw->vs.attribute_bufs = cmdbuf->state.gfx.vs.attrib_bufs;
   draw->indirect_info.attribs = cmdbuf->state.gfx.vs.indirect_attribs_infos;
   draw->indirect_info.attrib_bufs =
      cmdbuf->state.gfx.vs.indirect_attrib_bufs_infos;
}

static void
panvk_emit_viewport(struct panvk_cmd_buffer *cmdbuf,
                    struct mali_viewport_packed *vpd)
{
   const struct vk_viewport_state *vp = &cmdbuf->vk.dynamic_graphics_state.vp;

   if (vp->viewport_count < 1)
      return;

   const VkViewport *viewport = &vp->viewports[0];
   const VkRect2D *scissor = &vp->scissors[0];
   float minz, maxz;
   panvk_depth_range(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state.vp,
                     &minz, &maxz);

   /* The spec says "width must be greater than 0.0" */
   assert(viewport->width >= 0);
   int minx = (int)viewport->x;
   int maxx = (int)(viewport->x + viewport->width);

   /* Viewport height can be negative */
   int miny = MIN2((int)viewport->y, (int)(viewport->y + viewport->height));
   int maxy = MAX2((int)viewport->y, (int)(viewport->y + viewport->height));

   assert(scissor->offset.x >= 0 && scissor->offset.y >= 0);
   minx = MAX2(scissor->offset.x, minx);
   miny = MAX2(scissor->offset.y, miny);
   maxx = MIN2(scissor->offset.x + scissor->extent.width, maxx);
   maxy = MIN2(scissor->offset.y + scissor->extent.height, maxy);

   /* Make sure we don't end up with a max < min when width/height is 0 */
   maxx = maxx > minx ? maxx - 1 : maxx;
   maxy = maxy > miny ? maxy - 1 : maxy;

   /* Clamp viewport scissor to valid range */
   minx = CLAMP(minx, 0, UINT16_MAX);
   maxx = CLAMP(maxx, 0, UINT16_MAX);
   miny = CLAMP(miny, 0, UINT16_MAX);
   maxy = CLAMP(maxy, 0, UINT16_MAX);

   pan_pack(vpd, VIEWPORT, cfg) {
      cfg.scissor_minimum_x = minx;
      cfg.scissor_minimum_y = miny;
      cfg.scissor_maximum_x = maxx;
      cfg.scissor_maximum_y = maxy;
      cfg.minimum_z = minz;
      cfg.maximum_z = maxz;
   }
}

static VkResult
panvk_draw_prepare_viewport(struct panvk_cmd_buffer *cmdbuf,
                            struct panvk_draw_data *draw)
{
   /* When rasterizerDiscardEnable is active, it is allowed to have viewport and
    * scissor disabled.
    * As a result, we define an empty one.
    */
   if (!cmdbuf->state.gfx.vpd || dyn_gfx_state_dirty(cmdbuf, VP_VIEWPORTS) ||
       dyn_gfx_state_dirty(cmdbuf, VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE) ||
       dyn_gfx_state_dirty(cmdbuf, VP_SCISSORS) ||
       dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_CLIP_ENABLE) ||
       dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_CLAMP_ENABLE) ||
       dyn_gfx_state_dirty(cmdbuf, VP_DEPTH_CLAMP_RANGE)) {
      struct pan_ptr vp = panvk_cmd_alloc_desc(cmdbuf, VIEWPORT);
      if (!vp.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      panvk_emit_viewport(cmdbuf, vp.cpu);
      cmdbuf->state.gfx.vpd = vp.gpu;
   }

   draw->viewport = cmdbuf->state.gfx.vpd;
   return VK_SUCCESS;
}

static void
panvk_emit_vertex_dcd(struct panvk_cmd_buffer *cmdbuf,
                      const struct panvk_draw_data *draw,
                      struct mali_draw_packed *dcd)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_desc_state *vs_desc_state =
      &cmdbuf->state.gfx.vs.desc;

   pan_pack(dcd, DRAW, cfg) {
      cfg.state = panvk_priv_mem_dev_addr(vs->rsd);
      cfg.attributes = draw->vs.attributes;
      cfg.attribute_buffers = draw->vs.attribute_bufs;
      cfg.varyings = draw->vs.varyings;
      cfg.varying_buffers = draw->varying_bufs;
      cfg.thread_storage = draw->tls;

      /* In case of indirect draw, the descriptor will be patched at runtime */
      if (!is_indirect_draw(draw)) {
         cfg.offset_start = draw->info.vertex.raw_offset;
         cfg.instance_size =
            draw->info.instance.count > 1 ? draw->padded_vertex_count : 1;
      }

      cfg.uniform_buffers = vs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_UBO];
      cfg.push_uniforms = cmdbuf->state.gfx.vs.push_uniforms;
      cfg.textures = vs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_TEXTURE];
      cfg.samplers = vs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_SAMPLER];
   }
}

static VkResult
panvk_draw_prepare_vertex_job(struct panvk_cmd_buffer *cmdbuf,
                              struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   struct pan_ptr ptr = panvk_cmd_alloc_desc(cmdbuf, COMPUTE_JOB);
   if (!ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   util_dynarray_append(&batch->jobs, ptr.cpu);
   draw->jobs.vertex = ptr;

   memcpy(pan_section_ptr(ptr.cpu, COMPUTE_JOB, INVOCATION), &draw->invocation,
          pan_size(INVOCATION));

   pan_section_pack(ptr.cpu, COMPUTE_JOB, PARAMETERS, cfg) {
      cfg.job_task_split = 5;
   }

   panvk_emit_vertex_dcd(cmdbuf, draw,
                         pan_section_ptr(ptr.cpu, COMPUTE_JOB, DRAW));
   return VK_SUCCESS;
}

static enum mali_draw_mode
translate_prim(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return MALI_DRAW_MODE_POINTS;
   case MESA_PRIM_LINES:
      return MALI_DRAW_MODE_LINES;
   case MESA_PRIM_LINE_STRIP:
      return MALI_DRAW_MODE_LINE_STRIP;
   case MESA_PRIM_TRIANGLES:
      return MALI_DRAW_MODE_TRIANGLES;
   case MESA_PRIM_TRIANGLE_STRIP:
      return MALI_DRAW_MODE_TRIANGLE_STRIP;
   case MESA_PRIM_TRIANGLE_FAN:
      return MALI_DRAW_MODE_TRIANGLE_FAN;
#if PAN_ARCH >= 9
   case MESA_PRIM_LINES_ADJACENCY:
      return MALI_DRAW_MODE_LINES_ADJACENCY;
   case MESA_PRIM_LINE_STRIP_ADJACENCY:
      return MALI_DRAW_MODE_LINE_STRIP_ADJACENCY;
   case MESA_PRIM_TRIANGLES_ADJACENCY:
      return MALI_DRAW_MODE_TRIANGLES_ADJACENCY;
   case MESA_PRIM_TRIANGLE_STRIP_ADJACENCY:
      return MALI_DRAW_MODE_TRIANGLE_STRIP_ADJACENCY;
#endif
   default:
      UNREACHABLE("Invalid primitive type");
   }
}

static void
panvk_emit_tiler_primitive(struct panvk_cmd_buffer *cmdbuf,
                           const struct panvk_draw_data *draw,
                           struct mali_primitive_packed *prim)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_input_assembly_state *ia = &dyns->ia;
   const struct vk_rasterization_state *rs = &dyns->rs;
   bool writes_point_size =
      vs->info.vs.writes_point_size &&
      ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   bool secondary_shader = vs->info.vs.secondary_enable && fs != NULL;
   assert(!(vs->info.outputs_written & VARYING_BIT_PRIMITIVE_ID));
   bool fs_reads_primitive_id = fs ? fs->info.fs.reads_primitive_id : false;

   pan_pack(prim, PRIMITIVE, cfg) {
      cfg.draw_mode = translate_prim(draw->info.prim);
      if (writes_point_size)
         cfg.point_size_array_format = MALI_POINT_SIZE_ARRAY_FORMAT_FP16;
      cfg.primitive_index_enable = fs_reads_primitive_id;
      cfg.primitive_index_writeback = fs_reads_primitive_id;

      cfg.first_provoking_vertex =
         cmdbuf->state.gfx.render.first_provoking_vertex != U_TRISTATE_NO;

      if (draw->info.index.restart_enable)
         cfg.primitive_restart = MALI_PRIMITIVE_RESTART_IMPLICIT;
      cfg.job_task_split = 6;

      if (draw->info.index.index_size) {
         switch (draw->info.index.index_size) {
         case 4:
            cfg.index_type = MALI_INDEX_TYPE_UINT32;
            break;
         case 2:
            cfg.index_type = MALI_INDEX_TYPE_UINT16;
            break;
         case 1:
            cfg.index_type = MALI_INDEX_TYPE_UINT8;
            break;
         default:
            UNREACHABLE("Invalid index size");
         }
      }

      /* In case of indirect draw, the descriptor will be patched at runtime */
      cfg.index_count = is_indirect_draw(draw) ? 1 : draw->info.vertex.count;

      cfg.low_depth_cull = cfg.high_depth_cull =
         vk_rasterization_state_depth_clip_enable(rs);

      cfg.secondary_shader = secondary_shader;
   }
}

static void
panvk_emit_tiler_primitive_size(struct panvk_cmd_buffer *cmdbuf,
                                const struct panvk_draw_data *draw,
                                struct mali_primitive_size_packed *primsz)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const bool writes_point_size =
      vs->info.vs.writes_point_size && draw->info.prim == MESA_PRIM_POINTS;

   pan_pack(primsz, PRIMITIVE_SIZE, cfg) {
      if (writes_point_size) {
         cfg.size_array = draw->psiz;
      } else {
         cfg.fixed_sized = draw->line_width;
      }
   }
}

static void
panvk_emit_tiler_dcd(struct panvk_cmd_buffer *cmdbuf,
                     const struct panvk_draw_data *draw,
                     struct mali_draw_packed *dcd)
{
   struct panvk_shader_desc_state *fs_desc_state = &cmdbuf->state.gfx.fs.desc;
   const struct vk_rasterization_state *rs =
      &cmdbuf->vk.dynamic_graphics_state.rs;

   enum mesa_prim reduced_prim = u_reduced_prim(draw->info.prim);
   const bool non_polygon = reduced_prim != MESA_PRIM_TRIANGLES;

   pan_pack(dcd, DRAW, cfg) {
      cfg.front_face_ccw = rs->front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE;
      cfg.cull_front_face =
         !non_polygon && (rs->cull_mode & VK_CULL_MODE_FRONT_BIT) != 0;
      cfg.cull_back_face =
         !non_polygon && (rs->cull_mode & VK_CULL_MODE_BACK_BIT) != 0;

      cfg.position = draw->position;
      cfg.state = draw->fs.rsd;
      cfg.attributes = fs_desc_state->img_attrib_table;
      cfg.attribute_buffers =
         fs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_IMG];
      cfg.viewport = draw->viewport;
      cfg.varyings = draw->fs.varyings;
      cfg.varying_buffers = cfg.varyings ? draw->varying_bufs : 0;
      cfg.thread_storage = draw->tls;

      /* For all primitives but lines DRAW.flat_shading_vertex must
       * be set to 0 and the provoking vertex is selected with the
       * PRIMITIVE.first_provoking_vertex field.
       */
      if (reduced_prim == MESA_PRIM_LINES)
         cfg.flat_shading_vertex = true;

      /* In case of indirect draw, the descriptor will be patched at runtime */
      if (!is_indirect_draw(draw)) {
         cfg.offset_start = draw->info.vertex.raw_offset;
         cfg.instance_size =
            draw->info.instance.count > 1 ? draw->padded_vertex_count : 1;
         uint32_t primitives_per_instance =
            DIV_ROUND_UP(draw->padded_vertex_count,
                         mesa_vertices_per_prim(draw->info.prim));
         /* instance_primitive_size has the same restrictions as
          * padded_vertex_count, so we can use pan_padded_vertex_count here. */
         cfg.instance_primitive_size =
            pan_padded_vertex_count(primitives_per_instance);
      }

      cfg.uniform_buffers = fs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_UBO];
      cfg.push_uniforms = cmdbuf->state.gfx.fs.push_uniforms;
      cfg.textures = fs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_TEXTURE];
      cfg.samplers = fs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_SAMPLER];

      cfg.occlusion_query = cmdbuf->state.gfx.occlusion_query.mode;
      cfg.occlusion = cmdbuf->state.gfx.occlusion_query.ptr;
#if PAN_ARCH == 9
      cfg.scissor_to_bounding_box = true;
#endif
   }
}

static void
set_provoking_vertex_mode(struct panvk_cmd_buffer *cmdbuf,
                          enum u_tristate first_provoking_vertex)
{
   struct panvk_cmd_graphics_state *state = &cmdbuf->state.gfx;

   if (first_provoking_vertex != U_TRISTATE_UNSET) {
      /* If this is not the first draw, first_provoking_vertex should match
       * the one from the previous draws. Unfortunately, we can't check it
       * when the render pass is inherited. */
      assert(state->render.first_provoking_vertex == U_TRISTATE_UNSET ||
             state->render.first_provoking_vertex == first_provoking_vertex);
      state->render.first_provoking_vertex = first_provoking_vertex;
   }

   /* Once we emit the first FBDs/TDs, we need to commit to a state. If we
    * choose the wrong one, we will fail the assert when the next application
    * draw happens (with a different state). Use PROVOKING_VERTEX_MODE_FIRST
    * because it's the vulkan default, and so likely to be right more often.
    *
    * TODO: handle this case better */
   if (state->render.first_provoking_vertex == U_TRISTATE_UNSET)
      state->render.first_provoking_vertex = U_TRISTATE_YES;
}

static VkResult
panvk_draw_prepare_tiler_job(struct panvk_cmd_buffer *cmdbuf,
                             struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   struct pan_ptr ptr;

   if (cmdbuf->state.gfx.fs.required) {
      const struct panvk_shader_desc_info *fs_desc_info =
         &cmdbuf->state.gfx.fs.shader->desc_info;
      struct panvk_shader_desc_state *fs_desc_state =
         &cmdbuf->state.gfx.fs.desc;
      VkResult result = panvk_per_arch(meta_get_copy_desc_job)(
         cmdbuf, fs_desc_info, &cmdbuf->state.gfx.desc_state,
         fs_desc_state, 0, &ptr);
      if (result != VK_SUCCESS)
         return result;
   }

   if (ptr.cpu)
      util_dynarray_append(&batch->jobs, ptr.cpu);

   draw->jobs.frag_copy_desc = ptr;

   ptr = panvk_cmd_alloc_desc(cmdbuf, TILER_JOB);
   util_dynarray_append(&batch->jobs, ptr.cpu);
   draw->jobs.tiler = ptr;

   memcpy(pan_section_ptr(ptr.cpu, TILER_JOB, INVOCATION), &draw->invocation,
          pan_size(INVOCATION));

   panvk_emit_tiler_primitive(cmdbuf, draw,
                              pan_section_ptr(ptr.cpu, TILER_JOB, PRIMITIVE));

   panvk_emit_tiler_primitive_size(
      cmdbuf, draw, pan_section_ptr(ptr.cpu, TILER_JOB, PRIMITIVE_SIZE));

   panvk_emit_tiler_dcd(cmdbuf, draw,
                        pan_section_ptr(ptr.cpu, TILER_JOB, DRAW));

   pan_section_pack(ptr.cpu, TILER_JOB, TILER, cfg) {
      cfg.address = PAN_ARCH >= 9 ? draw->tiler_ctx->valhall.desc
                                  : draw->tiler_ctx->bifrost.desc;
   }

   pan_section_pack(ptr.cpu, TILER_JOB, PADDING, padding)
      ;

   return VK_SUCCESS;
}

static VkResult
panvk_draw_prepare_idvs_job(struct panvk_cmd_buffer *cmdbuf,
                            struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   struct pan_ptr ptr = panvk_cmd_alloc_desc(cmdbuf, INDEXED_VERTEX_JOB);
   if (!ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   util_dynarray_append(&batch->jobs, ptr.cpu);
   draw->jobs.idvs = ptr;

   memcpy(pan_section_ptr(ptr.cpu, INDEXED_VERTEX_JOB, INVOCATION),
          &draw->invocation, pan_size(INVOCATION));

   panvk_emit_tiler_primitive(
      cmdbuf, draw, pan_section_ptr(ptr.cpu, INDEXED_VERTEX_JOB, PRIMITIVE));

   panvk_emit_tiler_primitive_size(
      cmdbuf, draw,
      pan_section_ptr(ptr.cpu, INDEXED_VERTEX_JOB, PRIMITIVE_SIZE));

   pan_section_pack(ptr.cpu, INDEXED_VERTEX_JOB, TILER, cfg) {
      cfg.address = PAN_ARCH >= 9 ? draw->tiler_ctx->valhall.desc
                                  : draw->tiler_ctx->bifrost.desc;
   }

   pan_section_pack(ptr.cpu, INDEXED_VERTEX_JOB, PADDING, _) {
   }

   panvk_emit_tiler_dcd(
      cmdbuf, draw,
      pan_section_ptr(ptr.cpu, INDEXED_VERTEX_JOB, FRAGMENT_DRAW));

   panvk_emit_vertex_dcd(
      cmdbuf, draw, pan_section_ptr(ptr.cpu, INDEXED_VERTEX_JOB, VERTEX_DRAW));
   return VK_SUCCESS;
}

static VkResult
panvk_draw_prepare_vs_copy_desc_job(struct panvk_cmd_buffer *cmdbuf,
                                    struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct panvk_shader_desc_info *vs_desc_info =
      &cmdbuf->state.gfx.vs.shader->desc_info;
   const struct panvk_shader_desc_state *vs_desc_state =
      &cmdbuf->state.gfx.vs.desc;
   const struct vk_vertex_input_state *vi =
      cmdbuf->vk.dynamic_graphics_state.vi;
   unsigned num_vbs = util_last_bit(vi->bindings_valid);
   struct pan_ptr ptr;

   VkResult result = panvk_per_arch(meta_get_copy_desc_job)(
      cmdbuf, vs_desc_info, &cmdbuf->state.gfx.desc_state, vs_desc_state,
      num_vbs * pan_size(ATTRIBUTE_BUFFER) * 2, &ptr);
   if (result != VK_SUCCESS)
      return result;

   if (ptr.cpu) {
      util_dynarray_append(&batch->jobs, ptr.cpu);
   }

   draw->jobs.vertex_copy_desc = ptr;
   return VK_SUCCESS;
}

static VkResult
panvk_draw_prepare_fs_copy_desc_job(struct panvk_cmd_buffer *cmdbuf,
                                    struct panvk_draw_data *draw)
{
   const struct panvk_shader_desc_info *fs_desc_info =
      &cmdbuf->state.gfx.fs.shader->desc_info;
   struct panvk_shader_desc_state *fs_desc_state = &cmdbuf->state.gfx.fs.desc;
   struct panvk_batch *batch = cmdbuf->cur_batch;
   struct pan_ptr ptr;

   VkResult result = panvk_per_arch(meta_get_copy_desc_job)(
      cmdbuf, fs_desc_info, &cmdbuf->state.gfx.desc_state,
      fs_desc_state, 0, &ptr);
   if (result != VK_SUCCESS)
      return result;

   if (ptr.cpu) {
      util_dynarray_append(&batch->jobs, ptr.cpu);
   }

   draw->jobs.frag_copy_desc = ptr;
   return VK_SUCCESS;
}

static VkResult
panvk_cmd_prepare_draw_link_shaders(struct panvk_cmd_buffer *cmd)
{
   struct panvk_cmd_graphics_state *gfx = &cmd->state.gfx;

   if (!gfx_state_dirty(cmd, VS) && !gfx_state_dirty(cmd, FS))
      return VK_SUCCESS;

   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmd->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmd));

   VkResult result =
      panvk_per_arch(link_shaders)(&cmd->desc_pool, vs, fs, &gfx->link);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return result;
   }

   return VK_SUCCESS;
}

static VkResult
prepare_draw(struct panvk_cmd_buffer *cmdbuf, struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   struct panvk_shader_desc_state *vs_desc_state = &cmdbuf->state.gfx.vs.desc;
   struct panvk_shader_desc_state *fs_desc_state = &cmdbuf->state.gfx.fs.desc;
   struct panvk_descriptor_state *desc_state = &cmdbuf->state.gfx.desc_state;
   const struct vk_rasterization_state *rs =
      &cmdbuf->vk.dynamic_graphics_state.rs;
   VkResult result;
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));

   /* There are only 16 bits in the descriptor for the job ID. Each job has a
    * pilot shader dealing with descriptor copies, and we need one
    * <vertex,tiler> pair per draw.
    */
   if (batch->vtc_jc.job_index + (4 * cmdbuf->state.gfx.render.layer_count) >=
       UINT16_MAX) {
      panvk_per_arch(cmd_close_batch)(cmdbuf);
      batch = panvk_per_arch(cmd_open_batch)(cmdbuf);
   }

   if (fs_user_dirty(cmdbuf)) {
      result = panvk_cmd_prepare_draw_link_shaders(cmdbuf);
      if (result != VK_SUCCESS)
         return result;
   }

   if (cmdbuf->state.gfx.vk_meta) {
      /* vk_meta doesn't care about the provoking vertex mode, we should use
       * the same mode that the application uses. */
      set_provoking_vertex_mode(cmdbuf, U_TRISTATE_UNSET);
   } else {
      enum u_tristate first_provoking_vertex = u_tristate_make(
         cmdbuf->vk.dynamic_graphics_state.rs.provoking_vertex ==
         VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT);
      set_provoking_vertex_mode(cmdbuf, first_provoking_vertex);
   }

   if (!rs->rasterizer_discard_enable) {
      ASSERTED const struct pan_fb_layout *fb =
         &cmdbuf->state.gfx.render.fb.layout;
      uint32_t *nr_samples = &cmdbuf->state.gfx.render.fb.nr_samples;
      uint32_t rasterization_samples =
         cmdbuf->vk.dynamic_graphics_state.ms.rasterization_samples;

      /* If there's no attachment, and the FB descriptor hasn't been allocated
       * yet, we patch nr_samples to match rasterization_samples, otherwise, we
       * make sure those two numbers match. */
      if (!batch->fb.desc.gpu && !cmdbuf->state.gfx.render.bound_attachments) {
         assert(rasterization_samples > 0);
         *nr_samples = rasterization_samples;
      } else {
         assert(rasterization_samples == *nr_samples);
      }

      /* In case we already emitted tiler/framebuffer descriptors, we ensure
       * that the sample count didn't change
       * XXX: This currently can happen in case we resume a render pass with no
       * attachements and without any draw as the FBD is emitted when suspending.
       */
      assert(fb->sample_count == 0 ||
             fb->sample_count == cmdbuf->state.gfx.render.fb.nr_samples);

      result = panvk_per_arch(cmd_alloc_fb_desc)(cmdbuf);
      if (result != VK_SUCCESS)
         return result;
   }

   panvk_per_arch(cmd_select_tile_size)(cmdbuf);

   result = panvk_per_arch(cmd_alloc_tls_desc)(cmdbuf, true);
   if (result != VK_SUCCESS)
      return result;

   const struct panvk_shader_desc_info *vs_desc_info =
      &cmdbuf->state.gfx.vs.shader->desc_info;
   const struct panvk_shader_desc_info *fs_desc_info =
      fs ? &cmdbuf->state.gfx.fs.shader->desc_info : NULL;

   uint32_t used_set_mask =
      vs_desc_info->used_set_mask | (fs ? fs_desc_info->used_set_mask : 0);

   if (gfx_state_dirty(cmdbuf, DESC_STATE) || gfx_state_dirty(cmdbuf, VS) ||
       gfx_state_dirty(cmdbuf, FS)) {
      result = panvk_per_arch(cmd_prepare_push_descs)(cmdbuf, desc_state,
                                                      used_set_mask);
      if (result != VK_SUCCESS)
         return result;
   }

   if (gfx_state_dirty(cmdbuf, DESC_STATE) || gfx_state_dirty(cmdbuf, VS)) {
      result = panvk_per_arch(cmd_prepare_shader_desc_tables)(
         cmdbuf, desc_state, vs_desc_info, false, vs_desc_state);
      if (result != VK_SUCCESS)
         return result;

      result = panvk_per_arch(cmd_prepare_dyn_ssbos)(
         cmdbuf, desc_state, vs_desc_info, vs_desc_state);
      if (result != VK_SUCCESS)
         return result;
   }

   /* This allocates and initializes the image table, which we need before we
    * can copy descriptors.
    */
   panvk_draw_prepare_attributes(cmdbuf, draw);

   if (gfx_state_dirty(cmdbuf, DESC_STATE) || gfx_state_dirty(cmdbuf, VS)) {
      result = panvk_draw_prepare_vs_copy_desc_job(cmdbuf, draw);
      if (result != VK_SUCCESS)
         return result;
   }

   if (gfx_state_dirty(cmdbuf, DESC_STATE) || gfx_state_dirty(cmdbuf, FS)) {
      if (fs == NULL) {
         /* No need to setup the FS desc tables if the FS is not executed. */
         memset(fs_desc_state, 0, sizeof(*fs_desc_state));
      } else {
         result = panvk_per_arch(cmd_prepare_shader_desc_tables)(
            cmdbuf, desc_state, fs_desc_info, true, fs_desc_state);
         if (result != VK_SUCCESS)
            return result;

         result = panvk_per_arch(cmd_prepare_dyn_ssbos)(
            cmdbuf, desc_state, fs_desc_info, fs_desc_state);
         if (result != VK_SUCCESS)
            return result;

         result = panvk_draw_prepare_fs_copy_desc_job(cmdbuf, draw);
         if (result != VK_SUCCESS)
            return result;
      }
   }

   draw->tls = batch->tls.gpu;
   draw->fb = batch->fb.desc.gpu;

   result = panvk_draw_prepare_fs_rsd(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return result;

   batch->tlsinfo.tls.size = MAX3(vs->info.tls_size, fs ? fs->info.tls_size : 0,
                                  batch->tlsinfo.tls.size);

   panvk_per_arch(cmd_prepare_draw_sysvals)(cmdbuf, &draw->info, fs);

   /* Viewport emission requires up-to-date {scale,offset}.z for min/max Z,
    * so we need to call it after calling cmd_prepare_draw_sysvals(), but
    * viewports are the same for all layers, so we only emit when layer_id=0.
    */
   result = panvk_draw_prepare_viewport(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return result;

   return VK_SUCCESS;
}

static VkResult
prepare_draw_layer(struct panvk_cmd_buffer *cmdbuf,
                   struct panvk_draw_data *draw, uint32_t layer)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   VkResult result;

   result = panvk_draw_prepare_varyings(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return result;

   draw->info.layer_id = layer;
   if (draw->info.layer_id > 0) {
      cmdbuf->state.gfx.sysvals.layer_id = draw->info.layer_id;
      gfx_state_set_dirty(cmdbuf, FS_PUSH_UNIFORMS);
   }

   struct pan_ptr vs_push_uniforms;
   result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(
      cmdbuf, vs, &vs_push_uniforms, 1);
   if (result != VK_SUCCESS)
      return result;
   cmdbuf->state.gfx.vs.push_uniforms = vs_push_uniforms.gpu;

   if (fs) {
      struct pan_ptr fs_push_uniforms;
      result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(
         cmdbuf, fs, &fs_push_uniforms, 1);
      if (result != VK_SUCCESS)
         return result;
      cmdbuf->state.gfx.fs.push_uniforms = fs_push_uniforms.gpu;
   }

   result = panvk_draw_prepare_tiler_context(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return result;

   if (vs->info.vs.idvs) {
      result = panvk_draw_prepare_idvs_job(cmdbuf, draw);
      if (result != VK_SUCCESS)
         return result;
   } else {
      result = panvk_draw_prepare_vertex_job(cmdbuf, draw);
      if (result != VK_SUCCESS)
         return result;

      bool needs_tiling =
         !cmdbuf->vk.dynamic_graphics_state.rs.rasterizer_discard_enable ||
         cmdbuf->state.gfx.occlusion_query.mode !=
            MALI_OCCLUSION_MODE_DISABLED;

      if (needs_tiling) {
         result = panvk_draw_prepare_tiler_job(cmdbuf, draw);
         if (result != VK_SUCCESS)
            return result;
      }
   }

   return VK_SUCCESS;
}

static void
panvk_cmd_draw(struct panvk_cmd_buffer *cmdbuf, struct panvk_draw_data *draw)
{
   const struct panvk_shader_variant *vs = panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   VkResult result;

   /* If there's no vertex shader, we can skip the draw. */
   if (!panvk_priv_mem_check_alloc(vs->rsd))
      return;

   /* Needs to be done before get_fs() is called because it depends on
    * fs.required being initialized. */
   cmdbuf->state.gfx.fs.required =
      fs_required(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state);

   result = prepare_draw(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return;

   pan_pack_work_groups_compute(&draw->invocation, 1, draw->vertex_range,
                                draw->info.instance.count, 1, 1, 1, true,
                                false);

   struct panvk_batch *batch = cmdbuf->cur_batch;

   unsigned copy_desc_job_id =
      draw->jobs.vertex_copy_desc.gpu
         ? pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, false, false,
                          0, 0, &draw->jobs.vertex_copy_desc, false)
         : 0;

   if (draw->jobs.frag_copy_desc.gpu) {
      /* We don't need to add frag_copy_desc as a dependency because the
       * tiler job doesn't execute the fragment shader, the fragment job
       * will, and the tiler/fragment synchronization happens at the batch
       * level. */
      pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, false, false, 0, 0,
                     &draw->jobs.frag_copy_desc, false);
   }

   uint32_t view_mask = cmdbuf->state.gfx.render.view_mask;
   assert(view_mask == 0 || util_bitcount(view_mask) <= batch->fb.layer_count);
   uint32_t enabled_layer_count = view_mask
                                     ? util_bitcount(view_mask)
                                     : cmdbuf->state.gfx.render.layer_count;

   for (uint32_t i = 0; i < enabled_layer_count; i++) {
      const uint32_t layer = (view_mask != 0) ? u_bit_scan(&view_mask) : i;
      result = prepare_draw_layer(cmdbuf, draw, layer);
      if (result != VK_SUCCESS)
         return;

      if (vs->info.vs.idvs) {
         pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_INDEXED_VERTEX, false,
                        false, 0, copy_desc_job_id, &draw->jobs.idvs, false);
      } else {
         unsigned vjob_id =
            pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_VERTEX, false, false,
                           0, copy_desc_job_id, &draw->jobs.vertex, false);

         if (draw->jobs.tiler.gpu != 0) {
            pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_TILER, false, false,
                           vjob_id, 0, &draw->jobs.tiler, false);
         }
      }
   }

   clear_dirty_after_draw(cmdbuf);
}

static void
panvk_cmd_draw_indirect(struct panvk_cmd_buffer *cmdbuf,
                        struct panvk_draw_data *draw)
{
   const struct panvk_shader_variant *vs = panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   VkResult result;

   /* If there's no vertex shader, we can skip the draw. */
   if (!panvk_priv_mem_check_alloc(vs->rsd))
      return;

   /* Needs to be done before get_fs() is called because it depends on
    * fs.required being initialized. */
   cmdbuf->state.gfx.fs.required =
      fs_required(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state);

   result = prepare_draw(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return;

   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct vk_vertex_input_state *vi =
      cmdbuf->vk.dynamic_graphics_state.vi;

   unsigned copy_desc_job_id =
      draw->jobs.vertex_copy_desc.gpu
         ? pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, false, false,
                          0, 0, &draw->jobs.vertex_copy_desc, false)
         : 0;

   if (draw->jobs.frag_copy_desc.gpu) {
      /* We don't need to add frag_copy_desc as a dependency because the
       * tiler job doesn't execute the fragment shader, the fragment job
       * will, and the tiler/fragment synchronization happens at the batch
       * level. */
      pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, false, false, 0, 0,
                     &draw->jobs.frag_copy_desc, false);
   }

   uint32_t view_mask = cmdbuf->state.gfx.render.view_mask;
   assert(view_mask == 0 || util_bitcount(view_mask) <= batch->fb.layer_count);
   uint32_t enabled_layer_count = view_mask
                                     ? util_bitcount(view_mask)
                                     : cmdbuf->state.gfx.render.layer_count;

   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   uint64_t index_min_max_res_ptr = 0;
   uint32_t job_before_indirect_helper = copy_desc_job_id;
   if (draw->info.index.index_size) {
      index_min_max_res_ptr =
         panvk_cmd_alloc_dev_mem(
            cmdbuf, desc,
            sizeof(struct libpan_draw_helper_index_min_max_result), 8)
            .gpu;
      const struct panlib_draw_index_minmax_search_helper_args args = {
         .index_buffer_ptr = draw->info.index.buffer_dev_addr,
         .cmd = draw->info.indirect.buffer_dev_addr,
         .min_ptr =
            index_min_max_res_ptr +
            offsetof(struct libpan_draw_helper_index_min_max_result, min),
         .max_ptr =
            index_min_max_res_ptr +
            offsetof(struct libpan_draw_helper_index_min_max_result, max),
      };

      struct libpan_draw_helper_index_min_max_result val = {
         .min = ((uint64_t)1 << (draw->info.index.index_size * 8)) - 1,
         .max = 0,
      };
      uint64_t *raw_val = (uint64_t *)&val;

      struct pan_ptr write_job =
         pan_pool_alloc_desc(&cmdbuf->desc_pool.base, WRITE_VALUE_JOB);

      pan_section_pack(write_job.cpu, WRITE_VALUE_JOB, PAYLOAD, payload) {
         payload.type = MALI_WRITE_VALUE_TYPE_IMMEDIATE_64;
         payload.address = index_min_max_res_ptr;
         payload.immediate_value = *raw_val;
      };

      unsigned write_job_id =
         pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_WRITE_VALUE, false, false,
                        0, copy_desc_job_id, &write_job, false);
      util_dynarray_append(&batch->jobs, write_job.cpu);

      const uint32_t index_count =
         draw->info.index.buffer_size / draw->info.index.index_size;
      uint32_t wg_count = DIV_ROUND_UP(index_count, 65536);
      assert(wg_count <= 65536);

      panlib_draw_index_minmax_search_helper_struct(
         &precomp_ctx, panlib_1d_with_jm_deps(wg_count, 0, write_job_id),
         PANLIB_BARRIER_NONE, args, util_logbase2(draw->info.index.index_size),
         draw->info.index.restart_enable);
      job_before_indirect_helper = batch->vtc_jc.job_index;
   }

   for (uint32_t i = 0; i < enabled_layer_count; i++) {
      const uint32_t layer = (view_mask != 0) ? u_bit_scan(&view_mask) : i;

      /* Force a new push uniform block to be allocated */
      gfx_state_set_dirty(cmdbuf, VS_PUSH_UNIFORMS);

      result = prepare_draw_layer(cmdbuf, draw, layer);
      if (result != VK_SUCCESS)
         return;

      assert(draw->info.indirect.buffer_dev_addr != 0 ||
             draw->info.index.index_size);

      uint32_t attrib_bufs_valid = vi->bindings_valid;
      uint32_t attribs_valid = vi->attributes_valid;
      uint64_t first_vertex_sysval = 0x8ull << 60;
      uint64_t first_instance_sysval = 0x8ull << 60;
      uint64_t raw_vertex_offset_sysval = 0x8ull << 60;
      if (shader_uses_sysval(vs, graphics, vs.first_vertex)) {
         first_vertex_sysval = cmdbuf->state.gfx.vs.push_uniforms +
                               shader_remapped_sysval_offset(
                                  vs, sysval_offset(graphics, vs.first_vertex));
      }

      if (shader_uses_sysval(vs, graphics, vs.base_instance)) {
         first_instance_sysval =
            cmdbuf->state.gfx.vs.push_uniforms +
            shader_remapped_sysval_offset(
               vs, sysval_offset(graphics, vs.base_instance));
      }

      if (shader_uses_sysval(vs, graphics, vs.raw_vertex_offset)) {
         raw_vertex_offset_sysval =
            cmdbuf->state.gfx.vs.push_uniforms +
            shader_remapped_sysval_offset(
               vs, sysval_offset(graphics, vs.raw_vertex_offset));
      }

      enum panlib_barrier indirect_barrier =
         PANLIB_BARRIER_JM_SUPPRESS_PREFETCH;
      struct panlib_precomp_grid indirect_grid =
         panlib_1d_with_jm_deps(1, 0, job_before_indirect_helper);

      if (draw->info.indirect.buffer_dev_addr != 0 &&
          draw->info.index.index_size) {
         const struct panlib_draw_indexed_indirect_helper_args args = {
            .cmd = draw->info.indirect.buffer_dev_addr,
            .index_buffer_ptr = draw->info.index.buffer_dev_addr,
            .index_min_max_res = index_min_max_res_ptr,
            .index_size = draw->info.index.index_size,
            .primitive_vertex_count =
               mesa_vertices_per_prim(draw->info.prim),
            .varying_bufs_descs = draw->varying_bufs,
            .varying_bufs_info = draw->indirect_info.varying_bufs,
            .attrib_bufs_descs = draw->vs.attribute_bufs,
            .attrib_bufs_infos = draw->indirect_info.attrib_bufs,
            .attrib_bufs_valid = attrib_bufs_valid,
            .attribs_valid = attribs_valid,
            .attribs_descs = draw->vs.attributes,
            .attribs_infos = draw->indirect_info.attribs,
            .first_vertex_sysval = first_vertex_sysval,
            .first_instance_sysval = first_instance_sysval,
            .raw_vertex_offset_sysval = raw_vertex_offset_sysval,
            .idvs_job = vs->info.vs.idvs ? draw->jobs.idvs.gpu : 0,
            .vertex_job = draw->jobs.vertex.gpu,
            .tiler_job = draw->jobs.tiler.gpu,
         };
         panlib_draw_indexed_indirect_helper_struct(&precomp_ctx, indirect_grid,
                                                    indirect_barrier, args);
      } else if (draw->info.indirect.buffer_dev_addr != 0) {
         const struct panlib_draw_indirect_helper_args args = {
            .cmd = draw->info.indirect.buffer_dev_addr,
            .primitive_vertex_count =
               mesa_vertices_per_prim(draw->info.prim),
            .varying_bufs_descs = draw->varying_bufs,
            .varying_bufs_info = draw->indirect_info.varying_bufs,
            .attrib_bufs_descs = draw->vs.attribute_bufs,
            .attrib_bufs_infos = draw->indirect_info.attrib_bufs,
            .attrib_bufs_valid = attrib_bufs_valid,
            .attribs_valid = attribs_valid,
            .attribs_descs = draw->vs.attributes,
            .attribs_infos = draw->indirect_info.attribs,
            .first_vertex_sysval = first_vertex_sysval,
            .first_instance_sysval = first_instance_sysval,
            .raw_vertex_offset_sysval = raw_vertex_offset_sysval,
            .idvs_job = vs->info.vs.idvs ? draw->jobs.idvs.gpu : 0,
            .vertex_job = draw->jobs.vertex.gpu,
            .tiler_job = draw->jobs.tiler.gpu,
         };
         panlib_draw_indirect_helper_struct(&precomp_ctx, indirect_grid,
                                            indirect_barrier, args);
      } else {
         assert(false && "Invalid indirect draw");
      }

      /* Grab the index of the indirect helper job */
      uint32_t prev_job = batch->vtc_jc.job_index;

      if (vs->info.vs.idvs) {
         pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_INDEXED_VERTEX, false,
                        false, 0, prev_job, &draw->jobs.idvs, false);
      } else {
         unsigned vjob_id =
            pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_VERTEX, false, true, 0,
                           prev_job, &draw->jobs.vertex, false);

         if (draw->jobs.tiler.gpu != 0) {
            pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_TILER, false, false,
                           vjob_id, 0, &draw->jobs.tiler, false);
         }
      }
   }

   /*
    * We split every ~1024 indirect draw.
    * This is here for multiple reasons:
    * - The indirect varying buffer offset need to be reset at some point to
    * avoid going outside of bounds.
    * - It is possible to always end up with timeouts for batches with 4k draws
    * (see "dEQP-VK.api.command_buffers.many_indirect_draws_on_secondary") At
    * the same time, because of how TLS works on Mali, we should not split too
    * much as this will cause the TLS budget to go crazy.
    */
   if (batch->vtc_jc.job_index > (5 * 1024)) {
      panvk_per_arch(cmd_close_batch)(cmdbuf);
      batch = panvk_per_arch(cmd_open_batch)(cmdbuf);
      cmdbuf->state.gfx.vs.indirect_varying_bufs_infos = 0;
   }

   clear_dirty_after_draw(cmdbuf);
}

static unsigned
padded_vertex_count(struct panvk_cmd_buffer *cmdbuf, uint32_t vertex_count,
                    uint32_t instance_count)
{
   if (instance_count == 1)
      return vertex_count;

   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   bool idvs = vs->info.vs.idvs;

   /* Index-Driven Vertex Shading requires different instances to
    * have different cache lines for position results. Each vertex
    * position is 16 bytes and the Mali cache line is 64 bytes, so
    * the instance count must be aligned to 4 vertices.
    */
   if (idvs)
      vertex_count = ALIGN_POT(vertex_count, 4);

   return pan_padded_vertex_count(vertex_count);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDraw)(VkCommandBuffer commandBuffer, uint32_t vertexCount,
                        uint32_t instanceCount, uint32_t firstVertex,
                        uint32_t firstInstance)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (instanceCount == 0 || vertexCount == 0)
      return;

   /* gl_BaseVertexARB is a signed integer, and it should expose the value of
    * firstVertex in a non-indexed draw. */
   assert(firstVertex < INT32_MAX);

   /* gl_BaseInstance is a signed integer, and it should expose the value of
    * firstInstnace. */
   assert(firstInstance < INT32_MAX);

   struct panvk_draw_data draw = {
      .info = {
         .vertex.base = firstVertex,
         .vertex.raw_offset = firstVertex,
         .vertex.count = vertexCount,
         .instance.base = firstInstance,
         .instance.count = instanceCount,
         .prim = panvk_get_client_prim(cmdbuf),
      },
      .vertex_range = vertexCount,
      .padded_vertex_count =
         padded_vertex_count(cmdbuf, vertexCount, instanceCount),
   };

   panvk_cmd_draw(cmdbuf, &draw);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndexed)(VkCommandBuffer commandBuffer,
                               uint32_t indexCount, uint32_t instanceCount,
                               uint32_t firstIndex, int32_t vertexOffset,
                               uint32_t firstInstance)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (instanceCount == 0 || indexCount == 0)
      return;

   /* gl_BaseInstance is a signed integer, and it should expose the value of
    * firstInstnace. */
   assert(firstInstance < INT32_MAX);

   struct pan_ptr indirect_index_alloc = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, sizeof(struct VkDrawIndexedIndirectCommand), 8);

   struct VkDrawIndexedIndirectCommand *indirect_index_alloc_ptr =
      indirect_index_alloc.cpu;

   *indirect_index_alloc_ptr = (struct VkDrawIndexedIndirectCommand){
      .indexCount = indexCount,
      .instanceCount = instanceCount,
      .firstIndex = firstIndex,
      .vertexOffset = vertexOffset,
      .firstInstance = firstInstance,
   };

   struct panvk_draw_data draw = {
      .info = {
         .index = panvk_draw_info_index(cmdbuf, 0),
         .indirect.buffer_dev_addr = indirect_index_alloc.gpu,
         .indirect.draw_count = 1,
         .indirect.stride = 0,
         .prim = panvk_get_client_prim(cmdbuf),
      },
   };

   panvk_cmd_draw_indirect(cmdbuf, &draw);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndirect)(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                                VkDeviceSize offset, uint32_t drawCount,
                                uint32_t stride)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   VK_FROM_HANDLE(panvk_buffer, buffer, _buffer);

   if (drawCount == 0)
      return;

   /* We cannot support arbitrary draw count on JM */
   assert(drawCount == 1);

   struct panvk_draw_data draw = {
      .info = {
         .indirect.buffer_dev_addr = panvk_buffer_gpu_ptr(buffer, offset),
         .indirect.draw_count = drawCount,
         .indirect.stride = stride,
         .prim = panvk_get_client_prim(cmdbuf),
      },
   };

   panvk_cmd_draw_indirect(cmdbuf, &draw);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndexedIndirect)(VkCommandBuffer commandBuffer,
                                       VkBuffer _buffer, VkDeviceSize offset,
                                       uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   VK_FROM_HANDLE(panvk_buffer, buffer, _buffer);

   /* Because we don't currently advertise nullDescriptor for JM, it is only
    * valid to draw with a null index buffer if the draw accesses 0 indices.
    * For direct draws, this is covered by checks on instancedCount and
    * indexCount. For indirect draws we need to add an additional check, under
    * the assumption that if the index buffer is null, the draw must be empty.
    */
   if (drawCount == 0 || cmdbuf->state.gfx.ib.size == 0)
      return;

   /* We cannot support arbitrary draw count on JM */
   assert(drawCount == 1);

   struct panvk_draw_data draw = {
      .info = {
         .index = panvk_draw_info_index(cmdbuf, 0),
         .indirect.buffer_dev_addr = panvk_buffer_gpu_ptr(buffer, offset),
         .indirect.draw_count = drawCount,
         .indirect.stride = stride,
         .prim = panvk_get_client_prim(cmdbuf),
      },
   };

   panvk_cmd_draw_indirect(cmdbuf, &draw);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBeginRendering)(VkCommandBuffer commandBuffer,
                                  const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   struct panvk_cmd_graphics_state *state = &cmdbuf->state.gfx;
   bool resuming = pRenderingInfo->flags & VK_RENDERING_RESUMING_BIT;

   /* When resuming from a suspended pass, the state should be unchanged. */
   if (resuming && cmdbuf->cur_batch) {
      state->render.flags = pRenderingInfo->flags;
   } else {
      /* If we're not resuming, cur_batch should be NULL.  However, this
       * currently isn't true because of how events are implemented.
       *
       * XXX: Rewrite events to not close and open batch and add an assert here.
       */
      if (cmdbuf->cur_batch)
         panvk_per_arch(cmd_close_batch)(cmdbuf);

      panvk_per_arch(cmd_init_render_state)(cmdbuf, pRenderingInfo);
      cmdbuf->state.gfx.render.fb.needs_load = !resuming;
   }

   if (!cmdbuf->cur_batch)
      panvk_per_arch(cmd_open_batch)(cmdbuf);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdEndRendering2KHR)(
   VkCommandBuffer commandBuffer,
   UNUSED const VkRenderingEndInfoKHR *pRenderingEndInfo)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (!(cmdbuf->state.gfx.render.flags & VK_RENDERING_SUSPENDING_BIT)) {
      const struct pan_fb_load *fb_load = &cmdbuf->state.gfx.render.fb.load;
      bool always_load = fb_load->z.always || fb_load->s.always;
      for (unsigned rt = 0; rt < PAN_MAX_RTS; rt++) {
         if (fb_load->rts[rt].always)
            always_load = true;
      }

      if (always_load)
         panvk_per_arch(cmd_alloc_fb_desc)(cmdbuf);

      cmdbuf->state.gfx.render.fb.needs_store = true;

      panvk_per_arch(cmd_close_batch)(cmdbuf);
      cmdbuf->cur_batch = NULL;
      panvk_per_arch(cmd_meta_resolve_attachments)(cmdbuf);
   }
}

#else /* PAN_ARCH >= 9 -- lihat catatan PATCH di awal file */

#include "pan_encoder.h" /* PATCH: aslinya cuma ke-include dalam blok <9 */

/* PATCH v9 -- BAGIAN 1: disalin verbatim dari csf/panvk_vX_cmd_draw.c
 * (v10+, production-tested). Dikonfirmasi 100% CPU-only, nol dependensi
 * cs_* (command-stream builder CSF-only) -- portable ke JM/v9 tanpa
 * modifikasi struktural. Belum dipanggil dari mana pun (CmdDraw masih
 * stub di bawah) -- ini baru memvalidasi bagian ini compile bersih
 * berdiri sendiri, sebelum disambung ke job encoding draw yang sebenarnya. */

static void
emit_vs_attrib(struct panvk_cmd_buffer *cmdbuf,
               uint32_t attrib_idx, uint32_t vb_desc_offset,
               struct mali_attribute_packed *desc)
{
   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_vertex_input_state *vi = dyns->vi;
   const struct vk_vertex_attribute_state *attrib_info =
      &vi->attributes[attrib_idx];
   const struct vk_vertex_binding_state *buf_info =
      &vi->bindings[attrib_info->binding];
   const uint32_t stride = dyns->vi_binding_strides[attrib_info->binding];
   bool per_instance = buf_info->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
   enum pipe_format f = vk_format_to_pipe_format(attrib_info->format);
   unsigned buf_idx = vb_desc_offset + attrib_info->binding;

   pan_pack(desc, ATTRIBUTE, cfg) {
      cfg.offset = attrib_info->offset;

      if (per_instance)
         cfg.offset += cmdbuf->state.gfx.vi.base_instance * stride;

      cfg.format = GENX(pan_format_from_pipe_format)(f)->hw;
      cfg.table = 0;
      cfg.buffer_index = buf_idx;
      cfg.stride = stride;
      if (!per_instance) {
         /* Per-vertex */
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_VERTEX;
         cfg.offset_enable = true;
      } else if (buf_info->divisor == 1) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
      } else if (buf_info->divisor == 0) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         /* HW doesn't support a zero divisor, but we can achieve the same by
          * not using a divisor and setting the stride to zero */
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
         cfg.stride = 0;
      } else if (util_is_power_of_two_or_zero(buf_info->divisor)) {
         /* Per-instance, POT divisor */
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D_POT_DIVISOR;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
         cfg.divisor_r = __builtin_ctz(buf_info->divisor);
      } else {
         /* Per-instance, NPOT divisor */
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D_NPOT_DIVISOR;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
         cfg.divisor_d = pan_compute_npot_divisor(
            buf_info->divisor, &cfg.divisor_r, &cfg.divisor_e);
      }
   }
}

static VkResult
prepare_vs_driver_set(struct panvk_cmd_buffer *cmdbuf,
                      const struct panvk_shader *shader,
                      struct panvk_shader_desc_state *shader_desc_state,
                      uint32_t repeat_count)
{
   const struct panvk_shader_desc_info *vs_desc_info =
      &shader->desc_info;
   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_vertex_input_state *vi = dyns->vi;

   uint32_t vb_count = 0;
   u_foreach_bit(i, vi->attributes_valid)
      vb_count = MAX2(vi->attributes[i].binding + 1, vb_count);

   uint32_t vb_offset = vs_desc_info->dyn_bufs.count + MAX_VS_ATTRIBS + 1;
   uint32_t desc_count = vb_offset + vb_count;

   const struct panvk_descriptor_state *desc_state =
      &cmdbuf->state.gfx.desc_state;
   struct pan_ptr driver_set = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, repeat_count * desc_count * PANVK_DESCRIPTOR_SIZE,
      PANVK_DESCRIPTOR_SIZE);
   struct panvk_opaque_desc *descs = driver_set.cpu;

   if (!driver_set.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   for (uint32_t r = 0; r < repeat_count; r++) {
      for (uint32_t i = 0; i < MAX_VS_ATTRIBS; i++) {
         if (vi->attributes_valid & BITFIELD_BIT(i)) {
            emit_vs_attrib(cmdbuf, i, vb_offset,
                           (struct mali_attribute_packed *)(&descs[i]));
         } else {
            /* Write a NullDescriptor and rely on OOB behavior */
            pan_cast_and_pack(&descs[i], NULL_DESCRIPTOR, cfg)
               ;
         }
      }

      /* Dummy sampler always comes right after the vertex attribs. */
      pan_cast_and_pack(&descs[MAX_VS_ATTRIBS], SAMPLER, cfg) {
         cfg.clamp_integer_array_indices = false;
      }

      panvk_per_arch(cmd_fill_dyn_bufs)(
         desc_state, vs_desc_info,
         (struct mali_buffer_packed *)(&descs[MAX_VS_ATTRIBS + 1]));

      for (uint32_t i = 0; i < vb_count; i++) {
         const struct panvk_attrib_buf *vb = &cmdbuf->state.gfx.vb.bufs[i];
         const bool nulldesc = (vb->address == 0 && vb->size == 0);

         if ((vi->bindings_valid & BITFIELD_BIT(i)) && !nulldesc) {
            pan_cast_and_pack(&descs[vb_offset + i], BUFFER, cfg) {
               cfg.address = vb->address;
               cfg.size = vb->size;
            }
         } else {
            /* Write a NullDescriptor and rely on OOB behavior */
            pan_cast_and_pack(&descs[vb_offset + i], NULL_DESCRIPTOR, cfg)
               ;
         }
      }

      descs += desc_count;
   }

   shader_desc_state->driver_set.dev_addr = driver_set.gpu;
   shader_desc_state->driver_set.size = desc_count * PANVK_DESCRIPTOR_SIZE;
   gfx_state_set_dirty(cmdbuf, DESC_STATE);
   return VK_SUCCESS;
}

/* PATCH v9 -- BAGIAN 1b: sisi fragment dari driver set, plus pengisi
 * res_table.
 *
 * prepare_vs_driver_set() di atas hanya mengisi driver_set.{dev_addr,size}.
 * Yang dibaca job adalah res_table (lihat cfg.resources /
 * cfg.shader.resources di BAGIAN 2), dan res_table HANYA diisi oleh
 * panvk_per_arch(cmd_prepare_shader_res_table)(). Tanpa pemanggil itu,
 * res_table tetap 0 dan GPU membaca resource table dari alamat nol.
 *
 * emit_varying_descs() dan prepare_fs_driver_set() disalin verbatim dari
 * csf/panvk_vX_cmd_draw.c (v10+), sama seperti BAGIAN 1. Keduanya CPU-only,
 * nol dependensi cs_*. Layout driver set fragment dipaksa oleh
 * panvk_vX_nir_lower_descriptors.c (create_copy_table(): untuk stage
 * fragment dummy_sampler_idx = num_varying_attr_descs, lalu
 * dyn_bufs_start = dummy_sampler_idx + 1), jadi urutan
 * varyings -> dummy sampler -> dynamic buffers bukan pilihan bebas. */

static void
emit_varying_descs(const struct panvk_cmd_buffer *cmdbuf,
                   struct mali_attribute_packed *descs)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));

   const struct pan_varying_layout *vs_layout = &vs->info.varyings.formats;
   const struct pan_varying_layout *fs_format = &fs->info.varyings.formats;
   pan_varying_layout_require_layout(vs_layout);
   pan_varying_layout_require_format(fs_format);

   for (uint32_t i = 0; i < fs_format->count; i++) {
      const struct pan_varying_slot *fs_slot =
         pan_varying_layout_slot_at(fs_format, i);

      /* Skip empty slots and special varyings. */
      if (!fs_slot || fs_slot->section != PAN_VARYING_SECTION_GENERIC)
         continue;

      unsigned offset = 0;
      enum pipe_format format = PIPE_FORMAT_NONE;

      const struct pan_varying_slot *vs_slot =
         pan_varying_layout_find_slot(vs_layout, fs_slot->location);
      if (vs_slot) {
         nir_alu_type base_type = nir_alu_type_get_base_type(fs_slot->alu_type);
         nir_alu_type bit_size = nir_alu_type_get_type_size(vs_slot->alu_type);

         offset = vs_slot->offset;
         format = pan_varying_format(base_type | bit_size, vs_slot->ncomps);
      }

      pan_pack(&descs[i], ATTRIBUTE, cfg) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_VERTEX_PACKET;
         cfg.offset_enable = false;
         cfg.format = GENX(pan_format_from_pipe_format)(format)->hw;
         cfg.table = 61;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_VERTEX;
         cfg.offset = 1024 + offset;
         /* v9 has no hardware-controlled varying buffer index (that is v12+),
          * so the buffer index stays 0 here. */
         cfg.buffer_index = 0;
         cfg.attribute_stride = vs_layout->generic_size_B;
         cfg.packet_stride = vs_layout->generic_size_B + 16;
      }
   }
}

static VkResult
prepare_fs_driver_set(struct panvk_cmd_buffer *cmdbuf)
{
   const struct panvk_shader_desc_info *fs_desc_info =
      &cmdbuf->state.gfx.fs.shader->desc_info;
   struct panvk_shader_desc_state *fs_desc_state = &cmdbuf->state.gfx.fs.desc;
   const struct panvk_descriptor_state *desc_state =
      &cmdbuf->state.gfx.desc_state;
   /* If the shader is using LD_VAR_BUF[_IMM], we do not have to set up
    * Attribute Descriptors for varying loads. */
   const uint32_t desc_count = fs_desc_info->fs_varying_attr_desc_count +
                               fs_desc_info->dyn_bufs.count + 1;
   struct pan_ptr driver_set = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, desc_count * PANVK_DESCRIPTOR_SIZE, PANVK_DESCRIPTOR_SIZE);
   struct panvk_opaque_desc *descs = driver_set.cpu;

   if (desc_count && !driver_set.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   if (fs_desc_info->fs_varying_attr_desc_count > 0)
      emit_varying_descs(cmdbuf, (struct mali_attribute_packed *)(&descs[0]));

   /* Dummy sampler always comes right after the varyings. */
   const uint32_t sampler_idx = fs_desc_info->fs_varying_attr_desc_count;
   pan_cast_and_pack(&descs[sampler_idx], SAMPLER, cfg) {
      cfg.clamp_integer_array_indices = false;
   }

   panvk_per_arch(cmd_fill_dyn_bufs)(
      desc_state, fs_desc_info,
      (struct mali_buffer_packed *)(&descs[sampler_idx + 1]));

   fs_desc_state->driver_set.dev_addr = driver_set.gpu;
   fs_desc_state->driver_set.size = desc_count * PANVK_DESCRIPTOR_SIZE;
   gfx_state_set_dirty(cmdbuf, DESC_STATE);
   return VK_SUCCESS;
}

/*
 * Builds the driver set AND the resource table for both stages.
 *
 * repeat_count is 1 for the vertex stage: CSF keeps the base instance in a
 * command stream register and re-emits the descriptor when it changes, but JM
 * has no such register and cannot patch a job after it has been recorded, so
 * emit_vs_attrib() folds the base instance into the descriptor offset and the
 * driver set is rebuilt per draw instead of repeated.
 */
static VkResult
prepare_gfx_desc(struct panvk_cmd_buffer *cmdbuf)
{
   const struct panvk_shader *vs = cmdbuf->state.gfx.vs.shader;
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   const struct panvk_descriptor_state *desc_state =
      &cmdbuf->state.gfx.desc_state;
   struct panvk_shader_desc_state *vs_desc_state = &cmdbuf->state.gfx.vs.desc;
   struct panvk_shader_desc_state *fs_desc_state = &cmdbuf->state.gfx.fs.desc;
   VkResult result;

   result = prepare_vs_driver_set(cmdbuf, vs, vs_desc_state, 1);
   if (result != VK_SUCCESS)
      return result;

   /* Must come after the driver set: this is what fills res_table[0]. */
   result = panvk_per_arch(cmd_prepare_shader_res_table)(
      cmdbuf, desc_state, &vs->desc_info, vs_desc_state, 1);
   if (result != VK_SUCCESS)
      return result;

   if (fs == NULL) {
      /* No fragment shader means no fragment resources to point at. Zero it
       * so a stale res_table from an earlier draw is not reused. */
      memset(fs_desc_state, 0, sizeof(*fs_desc_state));
      return VK_SUCCESS;
   }

   result = prepare_fs_driver_set(cmdbuf);
   if (result != VK_SUCCESS)
      return result;

   return panvk_per_arch(cmd_prepare_shader_res_table)(
      cmdbuf, desc_state, &cmdbuf->state.gfx.fs.shader->desc_info,
      fs_desc_state, 1);
}

/* PATCH v9 -- BAGIAN 2: isi state vertex (POSITION, Shader Environment) dan
 * fragment (DRAW, struct Draw pendek v9) untuk Malloc Vertex Job.
 * ADAPTASI dari panvk_emit_vertex_dcd/panvk_emit_tiler_dcd Bifrost --
 * nama field genxml v9 dari riset kita sendiri (compute + genxml grep),
 * BELUM tervalidasi compile. Belum dipanggil dari mana pun. */

static void
panvk_emit_vs_position_v9(struct panvk_cmd_buffer *cmdbuf,
                          const struct panvk_draw_data *draw,
                          struct mali_shader_environment_packed *position_section)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_desc_state *vs_desc_state =
      &cmdbuf->state.gfx.vs.desc;

   pan_pack(position_section, SHADER_ENVIRONMENT, cfg) {
      cfg.resources = vs_desc_state->res_table;
   /* PATCH v9: vs->spd itu alias union ke vs->spds.pos_points (field
    * pertama union), BUKAN alamat generic yang aman dipakai selalu.
    * Harus pilih variant SPD sesuai topology, sama seperti csf/ (baris
    * ~1958): titik pakai pos_points, selain itu (termasuk segitiga)
    * pakai pos_triangles. Sebelum patch ini, draw non-titik salah baca
    * SPD dari pos_points yang kemungkinan besar tidak pernah
    * dialokasikan -> silent wrong render, bukan crash. */
   cfg.shader = draw->info.prim == MESA_PRIM_POINTS
                   ? panvk_priv_mem_dev_addr(vs->spds.pos_points)
                   : panvk_priv_mem_dev_addr(vs->spds.pos_triangles);
      cfg.thread_storage = draw->tls;
      cfg.fau = cmdbuf->state.gfx.vs.push_uniforms;
      cfg.fau_count = vs->fau.total_count;
   }
}

static enum mali_draw_mode
translate_prim(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return MALI_DRAW_MODE_POINTS;
   case MESA_PRIM_LINES:
      return MALI_DRAW_MODE_LINES;
   case MESA_PRIM_LINE_STRIP:
      return MALI_DRAW_MODE_LINE_STRIP;
   case MESA_PRIM_TRIANGLES:
      return MALI_DRAW_MODE_TRIANGLES;
   case MESA_PRIM_TRIANGLE_STRIP:
      return MALI_DRAW_MODE_TRIANGLE_STRIP;
   case MESA_PRIM_TRIANGLE_FAN:
      return MALI_DRAW_MODE_TRIANGLE_FAN;
   case MESA_PRIM_LINES_ADJACENCY:
      return MALI_DRAW_MODE_LINES_ADJACENCY;
   case MESA_PRIM_LINE_STRIP_ADJACENCY:
      return MALI_DRAW_MODE_LINE_STRIP_ADJACENCY;
   case MESA_PRIM_TRIANGLES_ADJACENCY:
      return MALI_DRAW_MODE_TRIANGLES_ADJACENCY;
   case MESA_PRIM_TRIANGLE_STRIP_ADJACENCY:
      return MALI_DRAW_MODE_TRIANGLE_STRIP_ADJACENCY;
   default:
      UNREACHABLE("Invalid primitive type");
   }
}

/* PATCH v9: duplikat standalone dari versi Bifrost (has_depth_att dkk,
 * jm/panvk_vX_cmd_draw.c baris ~104-180 di dalam blok #if PAN_ARCH<9).
 * Isinya murni logic VK-state, nol dependency hardware struct, aman
 * dipakai v9 apa adanya -- BUKAN ditulis ulang dari nol. */
static bool
has_depth_att_v9(struct panvk_cmd_buffer *cmdbuf)
{
   return (cmdbuf->state.gfx.render.bound_attachments &
           MESA_VK_RP_ATTACHMENT_DEPTH_BIT) != 0;
}

static bool
has_stencil_att_v9(struct panvk_cmd_buffer *cmdbuf)
{
   return (cmdbuf->state.gfx.render.bound_attachments &
           MESA_VK_RP_ATTACHMENT_STENCIL_BIT) != 0;
}

static bool
writes_depth_v9(struct panvk_cmd_buffer *cmdbuf)
{
   const struct vk_depth_stencil_state *ds =
      &cmdbuf->vk.dynamic_graphics_state.ds;

   return has_depth_att_v9(cmdbuf) && ds->depth.test_enable &&
          ds->depth.write_enable && ds->depth.compare_op != VK_COMPARE_OP_NEVER;
}

/* Copies of the Bifrost writes_stencil() / ds_test_always_passes() above,
 * needed by the v9 early-ZS selection in panvk_emit_fs_draw_v9(). */
static bool
writes_stencil_v9(struct panvk_cmd_buffer *cmdbuf)
{
   const struct vk_depth_stencil_state *ds =
      &cmdbuf->vk.dynamic_graphics_state.ds;

   return has_stencil_att_v9(cmdbuf) && ds->stencil.test_enable &&
          ((ds->stencil.front.write_mask &&
            (ds->stencil.front.op.fail != VK_STENCIL_OP_KEEP ||
             ds->stencil.front.op.pass != VK_STENCIL_OP_KEEP ||
             ds->stencil.front.op.depth_fail != VK_STENCIL_OP_KEEP)) ||
           (ds->stencil.back.write_mask &&
            (ds->stencil.back.op.fail != VK_STENCIL_OP_KEEP ||
             ds->stencil.back.op.pass != VK_STENCIL_OP_KEEP ||
             ds->stencil.back.op.depth_fail != VK_STENCIL_OP_KEEP)));
}

static bool
ds_test_always_passes_v9(struct panvk_cmd_buffer *cmdbuf)
{
   const struct vk_depth_stencil_state *ds =
      &cmdbuf->vk.dynamic_graphics_state.ds;

   if (!has_depth_att_v9(cmdbuf))
      return true;

   if (ds->depth.test_enable && ds->depth.compare_op != VK_COMPARE_OP_ALWAYS)
      return false;

   if (ds->stencil.test_enable &&
       (ds->stencil.front.op.compare != VK_COMPARE_OP_ALWAYS ||
        ds->stencil.back.op.compare != VK_COMPARE_OP_ALWAYS))
      return false;

   return true;
}

static inline enum mali_func
translate_compare_func_v9(VkCompareOp comp)
{
   return (enum mali_func)comp;
}

static enum mali_stencil_op
translate_stencil_op_v9(VkStencilOp in)
{
   switch (in) {
   case VK_STENCIL_OP_KEEP:
      return MALI_STENCIL_OP_KEEP;
   case VK_STENCIL_OP_ZERO:
      return MALI_STENCIL_OP_ZERO;
   case VK_STENCIL_OP_REPLACE:
      return MALI_STENCIL_OP_REPLACE;
   case VK_STENCIL_OP_INCREMENT_AND_CLAMP:
      return MALI_STENCIL_OP_INCR_SAT;
   case VK_STENCIL_OP_DECREMENT_AND_CLAMP:
      return MALI_STENCIL_OP_DECR_SAT;
   case VK_STENCIL_OP_INCREMENT_AND_WRAP:
      return MALI_STENCIL_OP_INCR_WRAP;
   case VK_STENCIL_OP_DECREMENT_AND_WRAP:
      return MALI_STENCIL_OP_DECR_WRAP;
   case VK_STENCIL_OP_INVERT:
      return MALI_STENCIL_OP_INVERT;
   default:
      UNREACHABLE("Invalid stencil op");
   }
}

static VkResult
panvk_emit_fs_draw_v9(struct panvk_cmd_buffer *cmdbuf,
                      const struct panvk_draw_data *draw,
                      struct mali_draw_packed *dcd)
{
   struct panvk_shader_desc_state *fs_desc_state = &cmdbuf->state.gfx.fs.desc;
   /* NULL when no fragment shader is needed (get_fs() / fs_required()). */
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   const struct vk_rasterization_state *rs =
      &cmdbuf->vk.dynamic_graphics_state.rs;
   const struct vk_depth_stencil_state *ds =
      &cmdbuf->vk.dynamic_graphics_state.ds;
   enum mesa_prim reduced_prim = u_reduced_prim(draw->info.prim);
   const bool non_polygon = reduced_prim != MESA_PRIM_TRIANGLES;
   /* PATCH v9: blend_ptr/bd_count SEKARANG udah disiapkan lebih awal di
    * prepare_draw_v9 (sebelum push-uniforms) - tinggal pakai, JANGAN
    * alokasi+isi ulang di sini (itu penyebab bug clear-only-nya). */
   uint32_t bd_count = draw->bd_count;
   struct pan_ptr blend_ptr = draw->blend_ptr;
   bool test_s = has_stencil_att_v9(cmdbuf) && ds->stencil.test_enable;
   bool test_z = has_depth_att_v9(cmdbuf) && ds->depth.test_enable;
   bool writes_z = writes_depth_v9(cmdbuf);

   struct pan_ptr ds_ptr = panvk_cmd_alloc_desc(cmdbuf, DEPTH_STENCIL);
   if (!ds_ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   float minz, maxz;
   panvk_depth_range(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state.vp,
                     &minz, &maxz);

   /* PATCH v9 Draw.depth_stencil: mapping field 1:1 dari referensi
    * panvk_draw_prepare_fs_rsd() (Bifrost RENDERER_STATE). 3 field
    * (depth_cull_enable, depth_clamp_mode, depth_source) TIDAK ADA
    * padanan eksplisit di kode Bifrost yang jadi referensi -- dibiarkan
    * default genxml (true / [0,1] / Fixed function), BUKAN tebakan. */
   pan_cast_and_pack(ds_ptr.cpu, DEPTH_STENCIL, cfg) {
      /* PATCH v9: depth source, stencil-from-shader, depth clip and clamp
       * were left at the genxml defaults (fixed-function depth, no shader
       * stencil, clip on, no clamp). With per-sample shading the
       * shader-written gl_FragDepth was then lost (CTS
       * depth_stencil_resolve.*, multisample.misc.*.ds_resolve_*), and
       * depthClamp / depthClipEnable were ignored. Same fields as CSF
       * prepare_ds(). PANVK_V9_ZSD_OLD=1 restores the old descriptor
       * (negative control). */
      static int zsd_old = -1;
      if (zsd_old < 0)
         zsd_old = pan_env_flag("PANVK_V9_ZSD_OLD") != NULL;
      if (!zsd_old) {
         cfg.stencil_from_shader = fs ? fs->info.fs.writes_stencil : 0;
         cfg.depth_cull_enable = vk_rasterization_state_depth_clip_enable(rs);
         if (rs->depth_clamp_enable)
            cfg.depth_clamp_mode = MALI_DEPTH_CLAMP_MODE_BOUNDS;
         if (fs)
            cfg.depth_source = pan_depth_source(&fs->info);
      }

      cfg.stencil_test_enable = test_s;
      cfg.depth_write_enable = writes_z;
      cfg.depth_bias_enable = rs->depth_bias.enable;
      cfg.depth_function =
         test_z ? translate_compare_func_v9(ds->depth.compare_op)
                : MALI_FUNC_ALWAYS;

      cfg.depth_units = rs->depth_bias.constant_factor;
      cfg.depth_factor = rs->depth_bias.slope_factor;
      cfg.depth_bias_clamp = rs->depth_bias.clamp;

      cfg.front_write_mask = ds->stencil.front.write_mask;
      cfg.back_write_mask = ds->stencil.back.write_mask;
      cfg.front_value_mask = ds->stencil.front.compare_mask;
      cfg.back_value_mask = ds->stencil.back.compare_mask;
      cfg.front_reference_value = ds->stencil.front.reference;
      cfg.back_reference_value = ds->stencil.back.reference;

      if (test_s) {
         cfg.front_compare_function =
            translate_compare_func_v9(ds->stencil.front.op.compare);
         cfg.front_stencil_fail =
            translate_stencil_op_v9(ds->stencil.front.op.fail);
         cfg.front_depth_fail =
            translate_stencil_op_v9(ds->stencil.front.op.depth_fail);
         cfg.front_depth_pass =
            translate_stencil_op_v9(ds->stencil.front.op.pass);
         cfg.back_compare_function =
            translate_compare_func_v9(ds->stencil.back.op.compare);
         cfg.back_stencil_fail =
            translate_stencil_op_v9(ds->stencil.back.op.fail);
         cfg.back_depth_fail =
            translate_stencil_op_v9(ds->stencil.back.op.depth_fail);
         cfg.back_depth_pass =
            translate_stencil_op_v9(ds->stencil.back.op.pass);
      }
   }

   pan_pack(dcd, DRAW, cfg) {
      if (rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM)
         cfg.flags_0.aligned_line_ends = true;

      cfg.flags_0.multisample_enable =
         cmdbuf->vk.dynamic_graphics_state.ms.rasterization_samples > 1 &&
         !(reduced_prim == MESA_PRIM_LINES &&
           rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM);

      cfg.flags_0.occlusion_query = cmdbuf->state.gfx.occlusion_query.mode;
      cfg.flags_0.scissor_to_bounding_box = true;

      cfg.flags_0.front_face_ccw = rs->front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE;

      /* The tessellator always emits triangles in the same order, the TES
       * winding (and a lower-left domain origin) is applied by flipping the
       * front face, like Honeykrisp does. Not valid once a GS sits between
       * TES and the rasterizer. */
      const struct panvk_cmd_graphics_state *tgfx = &cmdbuf->state.gfx;
      if (tgfx->tess.tcs && tgfx->tess.tes && !tgfx->gs.shader &&
          !pan_env_flag("PANVK_V9_TESS_NO_WINDING")) {
         struct panvk_tess_info tinfo = panvk_tess_info_merge(
            tgfx->tess.tcs->tess.info, tgfx->tess.tes->tess.info);
         cfg.flags_0.front_face_ccw ^= tinfo.ccw;
         cfg.flags_0.front_face_ccw ^=
            cmdbuf->vk.dynamic_graphics_state.ts.domain_origin ==
            VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
      }
      cfg.flags_0.cull_front_face =
         !non_polygon && (rs->cull_mode & VK_CULL_MODE_FRONT_BIT) != 0;
      cfg.flags_0.cull_back_face =
         !non_polygon && (rs->cull_mode & VK_CULL_MODE_BACK_BIT) != 0;

      cfg.occlusion = cmdbuf->state.gfx.occlusion_query.ptr;

      /* PATCH v9 Draw.flags_0 early-ZS: these were never set on v9, so the
       * hardware ran every draw as FORCE_EARLY (enum value 0): depth/stencil
       * update and occlusion counting happened before the fragment shader,
       * so samples killed by `discard` were still counted (CTS
       * occlusion_query.*precise*discard). Same selection as CSF
       * build_dcd_flags() and gallium jm_emit_draw(). FPK: see below.
       * PANVK_V9_DCD_NO_EARLYZS=1 restores the old flags
       * (negative control). */
      if (fs) {
         static int no_earlyzs = -1;
         if (no_earlyzs < 0)
            no_earlyzs = pan_env_flag("PANVK_V9_DCD_NO_EARLYZS") != NULL;

         const struct vk_dynamic_graphics_state *dyns =
            &cmdbuf->vk.dynamic_graphics_state;
         bool alpha_to_coverage = dyns->ms.alpha_to_coverage_enable;

         /* The clip/cull distance discard of the FS only runs when the
          * last pre-rasterization stage has distances. */
         const struct panvk_shader *last_vtx = cmdbuf->state.gfx.vs.shader;
         const bool clip_discard =
            fs->fs.clip_cull && last_vtx &&
            (last_vtx->clip_size || last_vtx->cull_size);

         if (!no_earlyzs) {
            enum pan_earlyzs_zs_tilebuf_read zs_read =
               (z_attachment_read(fs, &dyns->ial) ||
                s_attachment_read(fs, &dyns->ial))
                  ? PAN_EARLYZS_ZS_TILEBUF_READ_NO_OPT
                  : PAN_EARLYZS_ZS_TILEBUF_NOT_READ;
            bool oq = cmdbuf->state.gfx.occlusion_query.mode !=
                      MALI_OCCLUSION_MODE_DISABLED;
            struct pan_earlyzs_state earlyzs = pan_earlyzs_get(
               clip_discard ? fs->fs.earlyzs_lut_clip : fs->fs.earlyzs_lut,
               writes_z || writes_stencil_v9(cmdbuf) || oq, alpha_to_coverage,
               ds_test_always_passes_v9(cmdbuf), zs_read);

            cfg.flags_0.pixel_kill_operation = (enum mali_pixel_kill)earlyzs.kill;
            cfg.flags_0.zs_update_operation =
               (enum mali_pixel_kill)earlyzs.update;
            cfg.flags_0.shader_modifies_coverage =
               fs->info.fs.writes_coverage || fs->info.fs.can_discard ||
               clip_discard || alpha_to_coverage;
         }

         /* Forward pixel kill: a later opaque fragment may kill earlier ones
          * still in flight at the same pixel. Same rule as CSF
          * build_dcd_flags() and the v6/v7 RENDERER_STATE path above: only
          * when the shader can be killed safely (can_fpk), writes every bound
          * colour target, reads none of them back, nothing reads the
          * destination (blending), no alpha-to-coverage, no rasterization
          * order access. PANVK_V9_NO_FPK=1 keeps both bits off (the old,
          * conservative behaviour; A/B control). */
         static int no_fpk = -1;
         if (no_fpk < 0)
            no_fpk = getenv("PANVK_V9_NO_FPK") != NULL;
         if (!no_earlyzs && !no_fpk) {
            uint8_t rt_mask = cmdbuf->state.gfx.render.bound_attachments &
                              MESA_VK_RP_ATTACHMENT_ANY_COLOR_BITS;
            uint8_t rt_written = color_attachment_written_mask(fs, &dyns->cal);
            uint8_t rt_read = color_attachment_read_mask(fs, &dyns->ial, rt_mask);
            bool roa_color =
               dyns->rasterization_order_access & VK_IMAGE_ASPECT_COLOR_BIT;

            /* A target the shader writes but whose write mask is zero (or
             * whose colour writes are disabled) keeps its old contents, so
             * it is not "written": FPK would kill those contents. Partial
             * masks are already caught by any_dest_read
             * (pan_blend_reads_dest), a zero mask is not -- the CSF and
             * v6/v7 rules share that gap; gallium masks with the blend's
             * enabled_mask. Found by mrt_alias_test (RT1 write mask 0). */
            for (unsigned i = 0; i < MAX_RTS; i++) {
               if (!dyns->cb.attachments[i].write_mask ||
                   !(dyns->cb.color_write_enables & BITFIELD_BIT(i)))
                  rt_written &= ~BITFIELD_BIT(i);
            }

            cfg.flags_0.allow_forward_pixel_to_kill =
               fs->info.fs.can_fpk && !clip_discard && !(rt_mask & ~rt_written) &&
               !(rt_read & rt_written) && !alpha_to_coverage &&
               !cmdbuf->state.gfx.cb.info.any_dest_read && !roa_color;
            cfg.flags_0.allow_forward_pixel_to_be_killed =
               !fs->info.writes_global;
         }

         /* PATCH v9: per-sample shading was never requested on v9, so the
          * FS ran once per pixel even with sampleShadingEnable or a shader
          * that reads gl_SampleID / gl_SamplePosition / interpolateAtSample
          * (CTS multisample_shader_builtin.*, multisample_interpolation.*).
          * Same rule as CSF build_dcd_flags(): also when a blend shader runs
          * with MSAA (it writes one sample at a time).
          * PANVK_V9_NO_PER_SAMPLE=1 restores the old behaviour (control). */
         static int no_per_sample = -1;
         if (no_per_sample < 0)
            no_per_sample = pan_env_flag("PANVK_V9_NO_PER_SAMPLE") != NULL;
         if (!no_per_sample)
            cfg.flags_0.evaluate_per_sample =
               (fs->info.fs.sample_shading ||
                cmdbuf->state.gfx.cb.info.needs_shader) &&
               dyns->ms.rasterization_samples > 1;

         /* PATCH v9: alpha-to-coverage was never enabled in the DCD, so
          * alpha did not reduce coverage (CTS multisample.alpha_to_coverage.*
          * alpha_invisible / alpha_translucent). CSF sets the same bit.
          * Covered by the same control. */
         if (!no_per_sample)
            cfg.flags_0.alpha_to_coverage = alpha_to_coverage;
      } else {
         /* PATCH v9: no fragment shader (no colour attachment written and no
          * other FS side effect, e.g. depth-only passes). Same as gallium
          * jm_emit_draw() and CSF build_dcd_flags(). */
         cfg.flags_0.pixel_kill_operation = MALI_PIXEL_KILL_FORCE_EARLY;
         cfg.flags_0.zs_update_operation = MALI_PIXEL_KILL_FORCE_EARLY;
         cfg.flags_0.allow_forward_pixel_to_kill = true;
         cfg.flags_0.allow_forward_pixel_to_be_killed = true;
         cfg.flags_0.overdraw_alpha0 = true;
         cfg.flags_0.overdraw_alpha1 = true;
      }

      /* PATCH v9 Draw.flags_1: TIDAK ADA referensi Bifrost/GL langsung
       * (field ini spesifik Valhall, nol pemakaian di tree Mesa manapun).
       * Nilai dipetakan by-analogy dari konsep yang sama yang dipakai
       * Bifrost RENDERER_STATE (sample_mask dari dynamic state MSAA,
       * render_target_mask dari bound color attachments) -- BUKAN reuse
       * kode existing, karena strukturnya beda total. Paling perlu
       * dicurigai duluan kalau ada hasil aneh di rendering nanti. */
      cfg.flags_1.sample_mask = cmdbuf->vk.dynamic_graphics_state.ms.sample_mask;
      cfg.flags_1.render_target_mask =
         cmdbuf->state.gfx.render.bound_attachments &
         MESA_VK_RP_ATTACHMENT_ANY_COLOR_BITS;
      PAN_V9_TRACE( "[PANVK_DEBUG_FLAGS1] sample_mask=0x%x render_target_mask=0x%x bound_attachments=0x%x\n",
              cfg.flags_1.sample_mask, cfg.flags_1.render_target_mask,
              cmdbuf->state.gfx.render.bound_attachments);

      /* PATCH v9 Draw.vertex_array: genxml comment eksplisit bilang
       * Pointer/stride "Written by hardware in MallocVertexShader job
       * mode" -- CPU cuma perlu set packet=true, sisanya diisi hardware
       * otomatis. Belum ada 1 pun kode Mesa lain yang pernah nyentuh
       * field ini buat dijadiin referensi silang. */
      cfg.vertex_array.packet = true;

      cfg.minimum_z = minz;
      cfg.maximum_z = maxz;

      cfg.depth_stencil = ds_ptr.gpu;
      {
         PAN_V9_TRACE( "[PANVK_DEBUG_DS] 32 byte Depth/stencil desc:");
         uint8_t *dsb = (uint8_t *)ds_ptr.cpu;
         for (int i = 0; i < 32; i++) PAN_V9_TRACE( " %02x", dsb[i]);
         PAN_V9_TRACE( "\n");
         if (bd_count > 0) {
            PAN_V9_TRACE( "[PANVK_DEBUG_BLENDRAW] %u byte Blend desc(s):", bd_count * 16);
            uint8_t *bb = (uint8_t *)blend_ptr.cpu;
            for (uint32_t i = 0; i < bd_count * 16; i++) PAN_V9_TRACE( " %02x", bb[i]);
            PAN_V9_TRACE( "\n");
         }
      }

      cfg.blend_count = bd_count;
      cfg.blend = blend_ptr.gpu;

      /* PATCH v9: without a fragment shader there is no shader to point at
       * (get_fs() returned NULL and this dereferenced it: SIGSEGV in CTS
       * occlusion_query.no_attachments* / *no_color_attachments). Like
       * gallium jm_emit_draw(): no shader, no blend, no colour target. */
      if (fs) {
         cfg.shader.resources = fs_desc_state->res_table;
         cfg.shader.shader = panvk_priv_mem_dev_addr(fs->spd);
         cfg.shader.thread_storage = draw->tls;
         cfg.shader.fau = cmdbuf->state.gfx.fs.push_uniforms;
         cfg.shader.fau_count = fs->fau.total_count;
      } else {
         cfg.blend_count = 0;
         cfg.blend = 0;
         cfg.flags_1.render_target_mask = 0;
      }
   }

   return VK_SUCCESS;
}

/* PATCH v9 -- BAGIAN 3: job encoder + pemanggil.
 *
 * Menyambung BAGIAN 1/1b/2 ke MALLOC_VERTEX_JOB. Sebelum ini semua fungsi di
 * atas compile tapi nol pemanggil (dead code) dan CmdDraw* cuma stub kosong.
 *
 * v9 WAJIB pakai MALLOC_VERTEX_JOB, bukan COMPUTE_JOB+TILER_JOB (Bifrost) dan
 * bukan INDEXED_VERTEX_JOB. Dasarnya jm_launch_draw() di
 * src/gallium/drivers/panfrost/pan_jm.c:
 *
 *     #if PAN_ARCH == 9
 *        assert(idvs && "Memory allocated IDVS required on Valhall");
 *
 * yaitu satu-satunya jalur Valhall+JM yang ada in-tree. Layout section diambil
 * dari aggregate "Malloc Vertex Job" di genxml/v9.xml (size 384, 11 section).
 *
 * Yang BELUM ditangani di sini, sengaja, supaya unit ini tetap bisa
 * di-review/di-rollback sendiri:
 *  - indirect draw (CmdDrawIndirect* masih stub di bawah);
 *  - multi-layer/multiview: hanya layer 0 yang di-encode;
 *  - transform feedback dan tessellation.
 */

/* Same mapping as panfrost_translate_index_size() in
 * src/gallium/drivers/panfrost/pan_cmdstream.h. Duplicated rather than
 * included because that header pulls in the whole Gallium driver context. */
static enum mali_index_type
translate_index_size(unsigned size)
{
   STATIC_ASSERT(MALI_INDEX_TYPE_NONE == 0);
   STATIC_ASSERT(MALI_INDEX_TYPE_UINT8 == 1);
   STATIC_ASSERT(MALI_INDEX_TYPE_UINT16 == 2);

   return (size == 4) ? MALI_INDEX_TYPE_UINT32 : size;
}

/* set_provoking_vertex_mode() lives in the PAN_ARCH < 9 block, so it is not
 * visible from here. Same logic, kept separate so the Bifrost copy stays
 * untouched. */
static void
set_provoking_vertex_mode_v9(struct panvk_cmd_buffer *cmdbuf,
                             enum u_tristate first_provoking_vertex)
{
   struct panvk_cmd_graphics_state *state = &cmdbuf->state.gfx;

   if (first_provoking_vertex != U_TRISTATE_UNSET) {
      assert(state->render.first_provoking_vertex == U_TRISTATE_UNSET ||
             state->render.first_provoking_vertex == first_provoking_vertex);
      state->render.first_provoking_vertex = first_provoking_vertex;
   }

   /* Once the first FBDs/TDs are emitted we have to commit to a mode.
    * PROVOKING_VERTEX_MODE_FIRST is the Vulkan default, so it is right more
    * often. Same TODO as the Bifrost path: this deserves better handling. */
   if (state->render.first_provoking_vertex == U_TRISTATE_UNSET)
      state->render.first_provoking_vertex = U_TRISTATE_YES;
}

static VkResult
panvk_draw_prepare_malloc_vertex_job(struct panvk_cmd_buffer *cmdbuf,
                                     struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   const struct vk_input_assembly_state *ia =
      &cmdbuf->vk.dynamic_graphics_state.ia;
   const struct vk_rasterization_state *rs =
      &cmdbuf->vk.dynamic_graphics_state.rs;
   enum mesa_prim reduced_prim = u_reduced_prim(draw->info.prim);

   /* The varying ("secondary") shader only feeds the fragment shader, so it is
    * pointless without one. Mirrors jm_emit_malloc_vertex_job(). */
   const bool secondary_shader =
      fs != NULL && panvk_priv_mem_check_alloc(vs->spds.var);

   struct pan_ptr ptr = panvk_cmd_alloc_desc(cmdbuf, MALLOC_VERTEX_JOB);
   if (!ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   /* The descriptor pool hands back recycled memory, and not every byte of
    * the job is written below. Measured in CTS (asm_triangle then transfer
    * in one process): bytes +64..+111 and +320..+383 of this job still held
    * the previous user's clear colour (0xffbf4020), and the job ended with
    * DATA_INVALID_FAULT (0x58). Run alone, the same test got zeroed memory
    * and passed. Which descriptor fields those bytes belong to was not
    * established. Zero the whole job first. PANVK_MVJ_NO_ZERO=1 is the
    * control. */
   if (!pan_env_flag("PANVK_MVJ_NO_ZERO"))
      memset(ptr.cpu, 0, pan_size(MALLOC_VERTEX_JOB));

   util_dynarray_append(&batch->jobs, ptr.cpu);

   pan_section_pack(ptr.cpu, MALLOC_VERTEX_JOB, PRIMITIVE, cfg) {
      cfg.draw_mode = translate_prim(draw->info.prim);
      cfg.index_count = draw->info.vertex.count;

      /* EXPERIMENT: base_vertex_offset was hardcoded to 0 for non-indexed
       * draws, which made vkCmdDraw ignore firstVertex entirely. Feeding
       * vertex.base in both cases is the hypothesis under test. */
      cfg.base_vertex_offset = draw->info.vertex.base;

      if (draw->info.index.index_size) {
         cfg.index_type = translate_index_size(draw->info.index.index_size);
         cfg.base_vertex_offset = draw->info.vertex.base;
      }

      if (vs->info.vs.writes_point_size &&
          ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST)
         cfg.point_size_array_format = MALI_POINT_SIZE_ARRAY_FORMAT_FP16;

      cfg.primitive_restart = ia->primitive_restart_enable;
      /* PATCH v9: the depth cull bits were left at their default (on), so
       * with depthClamp (or depthClipEnable = false) the tiler still dropped
       * primitives outside [0, 1] (CTS draw.renderpass.depth_clamp.*_input_*).
       * Same as the Bifrost tiler descriptor. PANVK_V9_DEPTH_CULL_ALWAYS=1
       * keeps them on (control). */
      if (!pan_env_flag("PANVK_V9_DEPTH_CULL_ALWAYS"))
         cfg.low_depth_cull = cfg.high_depth_cull =
            vk_rasterization_state_depth_clip_enable(rs);
      /* PATCH v9: gl_PrimitiveID in the fragment shader. The primitive
       * index was never enabled, so the FS read 0, and the value a geometry
       * shader writes was ignored (CTS geometry.builtin_variable.*primitive_id*).
       * Same rule as CSF: enable when the FS reads it, override with the
       * value of the last pre-rasterization stage when that stage writes it.
       * PANVK_V9_NO_PRIM_ID=1 restores the old fields (control). */
      if (fs && fs->info.fs.reads_primitive_id &&
          !pan_env_flag("PANVK_V9_NO_PRIM_ID")) {
         cfg.primitive_index_enable = true;
         cfg.primitive_index_override =
            (vs->info.outputs_written & VARYING_BIT_PRIMITIVE_ID) != 0;
      }
      cfg.secondary_shader = secondary_shader;
   }

   pan_section_pack(ptr.cpu, MALLOC_VERTEX_JOB, INSTANCE_COUNT, cfg) {
      cfg.count = draw->info.instance.count;
   }

   pan_section_pack(ptr.cpu, MALLOC_VERTEX_JOB, ALLOCATION, cfg) {
      if (secondary_shader) {
         unsigned sz = vs->info.varyings.formats.generic_size_B;
         cfg.vertex_packet_stride = sz + 16;
         cfg.vertex_attribute_stride = sz;
      } else {
         /* Hardware requirement for "no varyings" */
         cfg.vertex_packet_stride = 16;
         cfg.vertex_attribute_stride = 0;
      }
   }

   pan_section_pack(ptr.cpu, MALLOC_VERTEX_JOB, TILER, cfg) {
      cfg.address = draw->tiler_ctx->valhall.desc;
   }

   /* The scissor is already folded into the viewport descriptor the Bifrost
    * path emits, but v9 carries it in the job itself. Reuse the same clamping
    * as panvk_emit_viewport() so both paths agree. */
   pan_section_pack(ptr.cpu, MALLOC_VERTEX_JOB, SCISSOR, cfg) {
      const struct vk_viewport_state *vp =
         &cmdbuf->vk.dynamic_graphics_state.vp;

      if (vp->scissor_count > 0) {
         /* multiViewport: scissor of the viewport this copy draws. */
         const VkRect2D *s =
            &vp->scissors[MIN2(draw->info.viewport_id, vp->scissor_count - 1)];
         uint32_t minx = s->offset.x;
         uint32_t miny = s->offset.y;
         uint32_t maxx = s->offset.x + s->extent.width;
         uint32_t maxy = s->offset.y + s->extent.height;

         /* Maximum is inclusive. */
         maxx = maxx > minx ? maxx - 1 : maxx;
         maxy = maxy > miny ? maxy - 1 : maxy;

         cfg.scissor_minimum_x = MIN2(minx, UINT16_MAX);
         cfg.scissor_minimum_y = MIN2(miny, UINT16_MAX);
         cfg.scissor_maximum_x = MIN2(maxx, UINT16_MAX);
         cfg.scissor_maximum_y = MIN2(maxy, UINT16_MAX);
      } else {
         cfg.scissor_maximum_x = MAX_FRAMEBUFFER_DIMENSION - 1;
         cfg.scissor_maximum_y = MAX_FRAMEBUFFER_DIMENSION - 1;
      }
   }

   pan_section_pack(ptr.cpu, MALLOC_VERTEX_JOB, PRIMITIVE_SIZE, cfg) {
      if (vs->info.vs.writes_point_size &&
          ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST)
         cfg.size_array = draw->psiz;
      else if (reduced_prim == MESA_PRIM_LINES)
         cfg.fixed_sized = rs->line.width;
      else
         cfg.fixed_sized = 1.0f;
   }

   pan_section_pack(ptr.cpu, MALLOC_VERTEX_JOB, INDICES, cfg) {
      /* The hardware starts fetching at this address, so firstIndex has to be
       * folded in here. Passing the buffer base alone made vkCmdDrawIndexed
       * ignore firstIndex entirely. info.index.offset is in index units, not
       * bytes, so it is scaled by the index size. */
      cfg.address = draw->info.index.buffer_dev_addr +
                    (uint64_t)draw->info.index.offset *
                       draw->info.index.index_size;
   }

   VkResult result = panvk_emit_fs_draw_v9(
      cmdbuf, draw, pan_section_ptr(ptr.cpu, MALLOC_VERTEX_JOB, DRAW));
   if (result != VK_SUCCESS)
      return result;

   panvk_emit_vs_position_v9(
      cmdbuf, draw, pan_section_ptr(ptr.cpu, MALLOC_VERTEX_JOB, POSITION));

   /* VARYING is left zeroed when there is no secondary shader. Gallium does
    * the same (`if (!secondary_shader) continue;`) and points the varying
    * environment at the same state as the position one otherwise. */
   if (secondary_shader) {
      pan_section_pack(ptr.cpu, MALLOC_VERTEX_JOB, VARYING, cfg) {
         const struct panvk_shader_desc_state *vs_desc_state =
            &cmdbuf->state.gfx.vs.desc;

         cfg.resources = vs_desc_state->res_table;
         cfg.shader = panvk_priv_mem_dev_addr(vs->spds.var);
         cfg.thread_storage = draw->tls;
         cfg.fau = cmdbuf->state.gfx.vs.push_uniforms;
         /* The FAU count was missing here while the POSITION environment sets
          * it, so the varying shader ran with a valid FAU pointer and a count
          * of zero. Gallium sets the count for both environments (pan_jm.c,
          * jm_emit_shader_env). Measured effect without it: every vertex
          * shader whose FAU block was 4 words or more produced a fully black
          * image, while the 3-word single-varying shaders used earlier in
          * Phase 4 were unaffected. Exactly which FAU word the varying shader
          * needs was not established. The varying shader shares the position
          * shader's push-uniform block, so the count is the same.
          * PANVK_VARYING_NO_FAU_COUNT reproduces the old behaviour. */
         if (!pan_env_flag("PANVK_VARYING_NO_FAU_COUNT"))
            cfg.fau_count = vs->fau.total_count;
      }
   }

   draw->jobs.idvs = ptr;
   return VK_SUCCESS;
}

/* PATCH DEBUG v9 (sementara) -- dump mentah Malloc Vertex Job, biar dicocokin
 * manual ke layout genxml. HAPUS setelah bug ketemu.
 *
 * WAJIB dipanggil SESUDAH pan_jc_add_job(): bytes +000..+031 itu Job Header,
 * dan yang menulisnya adalah pan_jc_add_job(), bukan
 * panvk_draw_prepare_malloc_vertex_job(). Dump sebelum itu selalu memberi
 * header nol -- yang berarti Type=0 (Not started), bukan 11 (Malloc vertex) --
 * dan itu artefak waktu pengambilan dump, bukan bug.
 *
 * Field yang paling penting dicek di sini (v9.xml "Job Header"):
 *   Type            word 4, bit 1..7    -> harus 11 (Malloc vertex)
 *   Index           word 4, bit 16..31  -> nomor job, bukan 0
 *   Dependency 1/2  word 5
 *   Next            word 6..7           -> 0 kalau job terakhir di chain
 */
static void
panvk_debug_dump_malloc_vertex_job(const struct pan_ptr *job, const char *label)
{
   if (!pan_v9_trace_enabled())
      return;

   const uint8_t *bytes = (const uint8_t *)job->cpu;
   FILE *out = stderr;
   uint32_t word4, word5;

   memcpy(&word4, bytes + 16, sizeof(word4));
   memcpy(&word5, bytes + 20, sizeof(word5));

   uint32_t type = (word4 >> 1) & BITFIELD_MASK(7);
   uint32_t index = word4 >> 16;

   fprintf(out,
           "[PANVK_DEBUG_MVJ] %s gpu=0x%" PRIx64
           " header: type=%u (%s) index=%u dep1=%u dep2=%u\n",
           label, job->gpu, type,
           type == MALI_JOB_TYPE_MALLOC_VERTEX ? "MALLOC_VERTEX == BENAR"
                                               : "BUKAN MALLOC_VERTEX",
           index, word5 & BITFIELD_MASK(16), word5 >> 16);

   fprintf(out, "[PANVK_DEBUG_MVJ] dump 384 byte job descriptor:\n");
   for (int row = 0; row < 384; row += 16) {
      fprintf(out, "[PANVK_DEBUG_MVJ] +%03d:", row);
      for (int col = 0; col < 16; col++)
         fprintf(out, " %02x", bytes[row + col]);
      fprintf(out, "\n");
   }
}

static VkResult
prepare_draw_v9(struct panvk_cmd_buffer *cmdbuf, struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   const struct vk_rasterization_state *rs =
      &cmdbuf->vk.dynamic_graphics_state.rs;
   VkResult result;

   /* Same job-index guard as the Bifrost path, minus the copy-descriptor
    * pilot job: v9 has none, and one draw is a single MALLOC_VERTEX job.
    * PANVK_V9_SPLIT_AT=<n> lowers the limit (test knob: forces batch splits
    * inside a render pass, so the spill/reload path can be checked with
    * small tests). */
   static int split_at = -1;
   if (split_at < 0)
      split_at = getenv("PANVK_V9_SPLIT_AT") ? atoi(getenv("PANVK_V9_SPLIT_AT")) : UINT16_MAX;
   if (batch->vtc_jc.job_index + cmdbuf->state.gfx.render.layer_count >=
       (unsigned)split_at) {
      panvk_per_arch(cmd_close_batch)(cmdbuf);
      batch = panvk_per_arch(cmd_open_batch)(cmdbuf);
   }

   /* PATCH v9: variableMultisampleRate. Without attachments the sample
    * count of the frame comes from the first draw; a draw with another
    * rasterizationSamples starts a new batch (nothing to load or store, so
    * the split is invisible). PANVK_V9_NO_VMSR=1 hides the feature
    * (control). */
   if (!rs->rasterizer_discard_enable &&
       !cmdbuf->state.gfx.render.bound_attachments && batch->fb.desc.gpu &&
       cmdbuf->vk.dynamic_graphics_state.ms.rasterization_samples !=
          cmdbuf->state.gfx.render.fb.nr_samples) {
      panvk_per_arch(cmd_close_batch)(cmdbuf);
      batch = panvk_per_arch(cmd_open_batch)(cmdbuf);
      /* The tile size depends on the sample count: pick it again
       * (cmd_select_tile_size). */
      cmdbuf->state.gfx.render.fb.layout.sample_count = 0;
   }

   if (cmdbuf->state.gfx.vk_meta) {
      /* vk_meta doesn't care about the provoking vertex mode, we should use
       * the same mode that the application uses. */
      set_provoking_vertex_mode_v9(cmdbuf, U_TRISTATE_UNSET);
   } else {
      set_provoking_vertex_mode_v9(
         cmdbuf, u_tristate_make(rs->provoking_vertex ==
                                 VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT));
   }

   if (!rs->rasterizer_discard_enable) {
      uint32_t *nr_samples = &cmdbuf->state.gfx.render.fb.nr_samples;
      uint32_t rasterization_samples =
         cmdbuf->vk.dynamic_graphics_state.ms.rasterization_samples;

      if (!batch->fb.desc.gpu && !cmdbuf->state.gfx.render.bound_attachments) {
         assert(rasterization_samples > 0);
         *nr_samples = rasterization_samples;
      } else {
         assert(rasterization_samples == *nr_samples);
      }

      result = panvk_per_arch(cmd_alloc_fb_desc)(cmdbuf);
      if (result != VK_SUCCESS)
         return result;
   }

   panvk_per_arch(cmd_select_tile_size)(cmdbuf);

   result = panvk_per_arch(cmd_alloc_tls_desc)(cmdbuf, true);
   if (result != VK_SUCCESS)
      return result;

   draw->tls = batch->tls.gpu;
   draw->fb = batch->fb.desc.gpu;

   const struct panvk_shader_desc_info *vs_desc_info =
      &cmdbuf->state.gfx.vs.shader->desc_info;
   const struct panvk_shader_desc_info *fs_desc_info =
      fs ? &cmdbuf->state.gfx.fs.shader->desc_info : NULL;
   uint32_t used_set_mask =
      vs_desc_info->used_set_mask | (fs ? fs_desc_info->used_set_mask : 0);

   result = panvk_per_arch(cmd_prepare_push_descs)(
      cmdbuf, &cmdbuf->state.gfx.desc_state, used_set_mask);
   if (result != VK_SUCCESS)
      return result;

   /* Driver sets + resource tables for both stages. */
   /* emit_vs_attrib() folds vi.base_instance * stride into every
    * instance-rate attribute offset, but nothing on the v9 path ever assigned
    * vi.base_instance, so firstInstance was ignored for instance-rate vertex
    * attributes even on direct draws. The CSF path sets it per draw
    * (csf/panvk_vX_cmd_draw.c). For indirect draws this is the placeholder 0
    * and the helper kernel adds the real firstInstance on the GPU. The VS
    * driver set is rebuilt on every v9 draw, so no dirty tracking is needed. */
   cmdbuf->state.gfx.vi.base_instance = draw->info.instance.base;

   result = prepare_gfx_desc(cmdbuf);
   if (result != VK_SUCCESS)
      return result;

   /* PATCH v9: siapin blend descriptor SEBELUM sysvals/push-uniforms,
    * biar cmdbuf->state.gfx.fs.blend_descs[] yang dibaca set_gfx_sysval()
    * di bawah ini udah final buat draw ini, bukan nilai draw sebelumnya. */
   if (fs) {
      draw->bd_count = cmdbuf->state.gfx.render.fb.layout.rt_count;
      if (draw->bd_count > 0) {
         draw->blend_ptr =
            panvk_cmd_alloc_desc_array(cmdbuf, draw->bd_count, BLEND);
         if (!draw->blend_ptr.gpu)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;

         result =
            panvk_per_arch(blend_emit_descs)(cmdbuf, draw->blend_ptr.cpu);
         if (result != VK_SUCCESS)
            return result;

         PAN_V9_TRACE( "[PANVK_DEBUG_BLENDFIX] blend_ptr.gpu=0x%lx bd_count=%u state.blend_descs[0]=0x%lx\n",
                 (unsigned long)draw->blend_ptr.gpu, draw->bd_count,
                 (unsigned long)cmdbuf->state.gfx.fs.blend_descs[0]);
      }
   }

   batch->tlsinfo.tls.size = MAX3(vs->info.tls_size, fs ? fs->info.tls_size : 0,
                                  batch->tlsinfo.tls.size);

   panvk_per_arch(cmd_prepare_draw_sysvals)(cmdbuf, &draw->info, fs);

   struct pan_ptr vs_push_uniforms;
   result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(cmdbuf, vs,
                                                          &vs_push_uniforms, 1);
   if (result != VK_SUCCESS)
      return result;
   cmdbuf->state.gfx.vs.push_uniforms = vs_push_uniforms.gpu;

   PAN_V9_TRACE( "[PANVK_DEBUG_VSFAU] gpu=0x%llx count=%u bytes=%u:",
           (unsigned long long)vs_push_uniforms.gpu,
           vs->fau.total_count, vs->fau.total_count * 8);
   for (uint32_t i = 0; i < vs->fau.total_count * 8; i++)
      PAN_V9_TRACE( " %02x", ((uint8_t *)vs_push_uniforms.cpu)[i]);
   PAN_V9_TRACE( "\n");

   if (fs) {
      struct pan_ptr fs_push_uniforms;
      result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(
         cmdbuf, fs, &fs_push_uniforms, 1);
      if (result != VK_SUCCESS)
         return result;
      cmdbuf->state.gfx.fs.push_uniforms = fs_push_uniforms.gpu;

      PAN_V9_TRACE( "[PANVK_DEBUG_FSFAU] gpu=0x%llx count=%u bytes=%u:",
              (unsigned long long)fs_push_uniforms.gpu,
              fs->fau.total_count, fs->fau.total_count * 8);
      for (uint32_t i = 0; i < fs->fau.total_count * 8; i++)
         PAN_V9_TRACE( " %02x", ((uint8_t *)fs_push_uniforms.cpu)[i]);
      PAN_V9_TRACE( "\n");
   }

   /* One tiler context per layer; panvk_cmd_draw_v9() calls this once per
    * layer (or view) with draw->info.layer_id set. */
   result = panvk_per_arch(cmd_prepare_tiler_context)(cmdbuf,
                                                      draw->info.layer_id);
   if (result != VK_SUCCESS)
      return result;

   draw->tiler_ctx = &batch->tiler.ctx;

   return panvk_draw_prepare_malloc_vertex_job(cmdbuf, draw);
}

static void panvk_cmd_draw_v9_layer(struct panvk_cmd_buffer *cmdbuf,
                                    struct panvk_draw_data *draw);

/* PATCH v9: transform feedback. The XFB variant of the vertex shader (built
 * with no IDVS, xfb outputs lowered to global stores) runs as one COMPUTE job
 * per draw in the batch's vertex/tiler chain, one invocation per vertex and
 * instance, like gallium jm_launch_xfb() on v9 and CSF launch_xfb(). The
 * job barrier orders it after the previous XFB job, then panlib_xfb_add
 * advances the per-buffer offsets on the GPU (CSF does that in the command
 * stream). Same limits as CSF: indirect draws are skipped, indexed draws
 * capture vertices in linear order. 🧪 EXPERIMENTAL. */
static VkResult
launch_xfb_v9(struct panvk_cmd_buffer *cmdbuf,
              const struct panvk_draw_data *draw)
{
   struct panvk_cmd_graphics_state *gfx = &cmdbuf->state.gfx;

   /* Only vertex shaders have an XFB variant (not the TES used as the
    * hardware VS when tessellation is active). */
   if (!gfx->vs.shader || gfx->vs.shader->vk.stage != MESA_SHADER_VERTEX)
      return VK_SUCCESS;

   const struct panvk_shader_variant *xfb =
      panvk_shader_xfb_variant(gfx->vs.shader);

   /* PANVK_V9_XFB_SKIP=1: no capture (negative control). */
   static int xfb_skip = -1;
   if (xfb_skip < 0)
      xfb_skip = pan_env_flag("PANVK_V9_XFB_SKIP") != NULL;

   if (xfb_skip || !gfx->xfb.enabled || !xfb || !gfx->xfb.offsets.gpu)
      return VK_SUCCESS;

   /* The variant only exists when the VS has xfb outputs. */
   if (!panvk_priv_mem_check_alloc(xfb->spds.pos_triangles))
      return VK_SUCCESS;

   if (draw->info.indirect.buffer_dev_addr)
      return VK_SUCCESS;

   const uint32_t n = draw->info.vertex.count;
   const uint32_t instance_count = draw->info.instance.count;
   if (!n || !instance_count)
      return VK_SUCCESS;

   /* XFB stores independent primitives. Lists only drop an incomplete last
    * primitive; strips, fans and adjacency topologies are captured linearly
    * into scratch memory and rewritten by panlib_xfb_expand. Points and
    * patches stay linear. PANVK_V9_XFB_NO_ADJ=1 keeps adjacency linear
    * (control). */
   int topo = -1;
   uint32_t out_count = n;
   switch (draw->info.prim) {
   case MESA_PRIM_LINES:
      out_count = n / 2 * 2;
      break;
   case MESA_PRIM_TRIANGLES:
      out_count = n / 3 * 3;
      break;
   case MESA_PRIM_LINE_STRIP:
      topo = PANLIB_XFB_LINE_STRIP;
      out_count = n >= 2 ? 2 * (n - 1) : 0;
      break;
   case MESA_PRIM_TRIANGLE_STRIP:
      topo = PANLIB_XFB_TRIANGLE_STRIP;
      out_count = n >= 3 ? 3 * (n - 2) : 0;
      break;
   case MESA_PRIM_TRIANGLE_FAN:
      topo = PANLIB_XFB_TRIANGLE_FAN;
      out_count = n >= 3 ? 3 * (n - 2) : 0;
      break;
   case MESA_PRIM_LINES_ADJACENCY:
      topo = PANLIB_XFB_LINE_LIST_ADJ;
      out_count = n / 4 * 2;
      break;
   case MESA_PRIM_LINE_STRIP_ADJACENCY:
      topo = PANLIB_XFB_LINE_STRIP_ADJ;
      out_count = n >= 4 ? 2 * (n - 3) : 0;
      break;
   case MESA_PRIM_TRIANGLES_ADJACENCY:
      topo = PANLIB_XFB_TRIANGLE_LIST_ADJ;
      out_count = n / 6 * 3;
      break;
   case MESA_PRIM_TRIANGLE_STRIP_ADJACENCY:
      topo = PANLIB_XFB_TRIANGLE_STRIP_ADJ;
      out_count = n >= 6 ? 3 * ((n - 4) / 2) : 0;
      break;
   default:
      break;
   }
   if (topo >= PANLIB_XFB_LINE_LIST_ADJ && pan_env_flag("PANVK_V9_XFB_NO_ADJ")) {
      topo = -1;
      out_count = n;
   }
   if (!out_count)
      return VK_SUCCESS;

   /* Vertices shaded per instance by the XFB job. */
   const uint32_t vertex_count = topo >= 0 ? n : out_count;

   struct panvk_batch *batch = cmdbuf->cur_batch;
   assert(batch);

   struct pan_ptr expand = {0};
   if (topo >= 0) {
      expand = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc, sizeof(struct panlib_xfb_expand_params), 64);
      if (!expand.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      struct panlib_xfb_expand_params *p = expand.cpu;
      memset(p, 0, sizeof(*p));

      uint64_t scratch_size = 0;
      for (uint32_t i = 0; i < PANVK_MAX_XFB_BUFFERS; i++)
         scratch_size += ALIGN_POT((uint64_t)vertex_count * instance_count *
                                      xfb->xfb_stride[i], 64);

      struct pan_ptr scratch =
         panvk_cmd_alloc_dev_mem(cmdbuf, varying, scratch_size, 64);
      if (!scratch.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      uint64_t addr = scratch.gpu;
      for (uint32_t i = 0; i < PANVK_MAX_XFB_BUFFERS; i++) {
         p->src[i] = addr;
         p->dst[i] = gfx->xfb.buffers[i].address;
         p->stride[i] = xfb->xfb_stride[i];
         addr += ALIGN_POT((uint64_t)vertex_count * instance_count *
                              xfb->xfb_stride[i], 64);
      }
      p->offsets = gfx->xfb.offsets.gpu;
      p->in_vertices = vertex_count;
      p->out_vertices = out_count;
      p->instances = instance_count;
      p->topology = topo;
      p->verts_per_prim = (topo == PANLIB_XFB_LINE_STRIP ||
                           topo == PANLIB_XFB_LINE_LIST_ADJ ||
                           topo == PANLIB_XFB_LINE_STRIP_ADJ)
                             ? 2
                             : 3;
      for (uint32_t i = 0; i < PANVK_MAX_XFB_BUFFERS; i++)
         p->size[i] = MIN2(gfx->xfb.buffers[i].size, UINT32_MAX);
   }

   for (uint32_t i = 0; i < PANVK_MAX_XFB_BUFFERS; i++) {
      if (topo >= 0) {
         const struct panlib_xfb_expand_params *p = expand.cpu;
         gfx->sysvals.xfb.base[i] = p->src[i];
         gfx->sysvals.xfb.offset_ptr[i] =
            expand.gpu + offsetof(struct panlib_xfb_expand_params, zero) +
            i * sizeof(uint32_t);
      } else {
         gfx->sysvals.xfb.base[i] = gfx->xfb.buffers[i].address;
         gfx->sysvals.xfb.offset_ptr[i] =
            gfx->xfb.offsets.gpu + i * sizeof(uint32_t);
      }
   }
   gfx->sysvals.xfb.num_vertices = vertex_count;

   struct pan_ptr xfb_push_uniforms;
   VkResult result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(
      cmdbuf, xfb, &xfb_push_uniforms, 1);
   if (result != VK_SUCCESS)
      return result;
   gfx->xfb.push_uniforms = xfb_push_uniforms.gpu;

   gfx->xfb.desc.driver_set = gfx->vs.desc.driver_set;
   result = panvk_per_arch(cmd_prepare_shader_res_table)(
      cmdbuf, &gfx->desc_state, &gfx->vs.shader->desc_info, &gfx->xfb.desc,
      1);
   if (result != VK_SUCCESS)
      return result;

   batch->tlsinfo.tls.size =
      MAX2(xfb->info.tls_size, batch->tlsinfo.tls.size);

   struct pan_ptr job = panvk_cmd_alloc_desc(cmdbuf, COMPUTE_JOB);
   if (!job.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   pan_section_pack(job.cpu, COMPUTE_JOB, PAYLOAD, cfg) {
      cfg.workgroup_size_x = 1;
      cfg.workgroup_size_y = 1;
      cfg.workgroup_size_z = 1;
      cfg.workgroup_count_x = vertex_count;
      cfg.workgroup_count_y = instance_count;
      cfg.workgroup_count_z = 1;
      cfg.compute.resources = gfx->xfb.desc.res_table;
      cfg.compute.shader = panvk_priv_mem_dev_addr(xfb->spds.pos_triangles);
      cfg.compute.thread_storage = draw->tls;
      cfg.compute.fau = gfx->xfb.push_uniforms;
      cfg.compute.fau_count = xfb->fau.total_count;
      cfg.compute.attribute_offset = draw->info.vertex.base;
      /* No barriers or shared memory in the XFB variant. */
      cfg.allow_merging_workgroups = true;
      cfg.task_increment = 1;
      cfg.task_axis = MALI_TASK_AXIS_Z;
   }

   util_dynarray_append(&batch->jobs, job.cpu);
   pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, true, false, 0, 0,
                  &job, false);

   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);

   if (topo >= 0)
      panlib_xfb_expand(&precomp_ctx, panlib_1d(out_count * instance_count),
                        PANLIB_BARRIER_JM_BARRIER, expand.gpu);

   uint32_t inc[PANVK_MAX_XFB_BUFFERS] = {0};
   for (uint32_t i = 0; i < PANVK_MAX_XFB_BUFFERS; i++)
      inc[i] = out_count * instance_count * xfb->xfb_stride[i];

   panlib_xfb_add(&precomp_ctx, panlib_1d(1), PANLIB_BARRIER_JM_BARRIER,
                  gfx->xfb.offsets.gpu, inc[0], inc[1], inc[2], inc[3]);

   return VK_SUCCESS;
}

/* PATCH v9: tessellation through libpoly, port of CSF launch_tess(). The
 * API vertex shader (SW variant) and the TCS run as COMPUTE jobs, then the
 * libpoly tessellator kernels count, prefix-sum and write the domain points
 * and a 32-bit index buffer into the poly heap, plus one
 * VkDrawIndexedIndirectCommand. The draw itself is an indexed indirect draw
 * with the TES as the hardware vertex shader. Every job carries the JM job
 * barrier, so each one waits for the previous ones; the draw's jobs come
 * after them in the same vertex/tiler chain. Like CSF, indirect draws with
 * tessellation are not supported. 🧪 EXPERIMENTAL. */
/* Build the COMPUTE job of a graphics-stage compute variant without queueing
 * it. With indirect, the grid is a placeholder that a helper kernel queued
 * before the job overwrites (panlib_gs_setup_indirect). job->gpu stays 0 for
 * an empty shader. */
static VkResult
prepare_gfx_cs_job_v9(struct panvk_cmd_buffer *cmdbuf,
                      const struct panvk_shader_variant *cs,
                      const struct panvk_shader_desc_state *desc,
                      uint64_t push_uniforms, uint32_t x, uint32_t y,
                      uint32_t z, bool indirect, struct pan_ptr *job_out)
{
   const struct pan_compute_dim dim = {x, y, z};

   *job_out = (struct pan_ptr){0};

   /* An empty shader (e.g. a vertex shader without outputs feeding the
    * TCS) has no code and no SPD. It has no side effects, skip it. */
   if (!cs->bin_size)
      return VK_SUCCESS;

   if (!panvk_priv_mem_check_alloc(cs->spd))
      return VK_ERROR_UNKNOWN;

   uint64_t tsd =
      panvk_per_arch(cmd_dispatch_prepare_tls)(cmdbuf, cs, &dim, indirect);
   if (!tsd)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct pan_ptr job = panvk_cmd_alloc_desc(cmdbuf, COMPUTE_JOB);
   if (!job.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   pan_section_pack(job.cpu, COMPUTE_JOB, PAYLOAD, cfg) {
      cfg.workgroup_size_x = cs->cs.local_size.x;
      cfg.workgroup_size_y = cs->cs.local_size.y;
      cfg.workgroup_size_z = cs->cs.local_size.z;
      cfg.workgroup_count_x = x;
      cfg.workgroup_count_y = y;
      cfg.workgroup_count_z = z;
      cfg.compute.resources = desc->res_table;
      cfg.compute.shader = panvk_priv_mem_dev_addr(cs->spd);
      cfg.compute.thread_storage = tsd;
      cfg.compute.fau = push_uniforms;
      cfg.compute.fau_count = cs->fau.total_count;
   }

   *job_out = job;
   return VK_SUCCESS;
}

/* Queue a job built by prepare_gfx_cs_job_v9(), after everything before it
 * in the chain. An indirect job is parked as NOT_STARTED until its helper
 * turns it into a COMPUTE (or NULL) job. */
static void
queue_gfx_cs_job_v9(struct panvk_cmd_buffer *cmdbuf, struct pan_ptr *job,
                    bool indirect)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;

   if (!job->gpu)
      return;

   util_dynarray_append(&batch->jobs, job->cpu);
   pan_jc_add_job(&batch->vtc_jc,
                  indirect ? MALI_JOB_TYPE_NOT_STARTED : MALI_JOB_TYPE_COMPUTE,
                  true, false, 0, 0, job, false);
}

static VkResult
launch_gfx_cs_v9(struct panvk_cmd_buffer *cmdbuf,
                 const struct panvk_shader_variant *cs,
                 const struct panvk_shader_desc_state *desc,
                 uint64_t push_uniforms, uint32_t x, uint32_t y, uint32_t z)
{
   struct pan_ptr job;
   VkResult result = prepare_gfx_cs_job_v9(cmdbuf, cs, desc, push_uniforms, x,
                                           y, z, false, &job);
   if (result != VK_SUCCESS)
      return result;

   queue_gfx_cs_job_v9(cmdbuf, &job, false);
   return VK_SUCCESS;
}

/* Reset the bump heap once per command buffer, before its first
 * tessellated or indirect GS draw (CSF reset_poly_heap_once). */
static void
reset_poly_heap_once_v9(struct panvk_cmd_buffer *cmdbuf)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);

   if (cmdbuf->state.uses_poly_heap)
      return;

   struct panvk_precomp_ctx ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   panlib_fill_scalar(&ctx, panlib_1d(1), PANLIB_BARRIER_JM_BARRIER,
                      dev->poly_heap->addr.dev +
                         offsetof(struct poly_heap, bottom),
                      0);
   cmdbuf->state.uses_poly_heap = true;
}

static VkResult
prepare_tess_descs_v9(struct panvk_cmd_buffer *cmdbuf)
{
   struct panvk_cmd_graphics_state *gfx = &cmdbuf->state.gfx;
   const struct panvk_shader *input_vs = gfx->tess.input_vs;
   const struct panvk_shader *tcs = gfx->tess.tcs;
   VkResult result;

   result = panvk_per_arch(cmd_prepare_push_descs)(
      cmdbuf, &gfx->desc_state,
      input_vs->desc_info.used_set_mask | tcs->desc_info.used_set_mask);
   if (result != VK_SUCCESS)
      return result;

   result = prepare_vs_driver_set(cmdbuf, input_vs, &gfx->tess.vs_desc, 1);
   if (result != VK_SUCCESS)
      return result;

   result = panvk_per_arch(cmd_prepare_shader_res_table)(
      cmdbuf, &gfx->desc_state, &input_vs->desc_info, &gfx->tess.vs_desc, 1);
   if (result != VK_SUCCESS)
      return result;

   /* The TCS runs as a compute shader: compute driver set layout (dummy
    * sampler, then dynamic buffers), as in jm/panvk_vX_cmd_dispatch.c. */
   const struct panvk_shader_desc_info *desc_info = &tcs->desc_info;
   const uint32_t desc_count = desc_info->dyn_bufs.count + 1;
   struct pan_ptr driver_set = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, desc_count * PANVK_DESCRIPTOR_SIZE, PANVK_DESCRIPTOR_SIZE);
   if (!driver_set.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct panvk_opaque_desc *descs = driver_set.cpu;
   pan_cast_and_pack(&descs[0], SAMPLER, cfg) {
      cfg.clamp_integer_array_indices = false;
   }
   panvk_per_arch(cmd_fill_dyn_bufs)(&gfx->desc_state, desc_info,
                                     (struct mali_buffer_packed *)&descs[1]);
   gfx->tess.tcs_desc.driver_set.dev_addr = driver_set.gpu;
   gfx->tess.tcs_desc.driver_set.size = desc_count * PANVK_DESCRIPTOR_SIZE;

   return panvk_per_arch(cmd_prepare_shader_res_table)(
      cmdbuf, &gfx->desc_state, &tcs->desc_info, &gfx->tess.tcs_desc, 1);
}

static void
launch_tessellator_v9(struct panvk_cmd_buffer *cmdbuf, uint32_t patch_count,
                      enum tess_primitive_mode mode, uint64_t params,
                      enum poly_tess_mode tess_mode)
{
   struct panvk_precomp_ctx ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   struct panlib_precomp_grid grid = panlib_1d(DIV_ROUND_UP(patch_count, 64));

   switch (mode) {
   case TESS_PRIMITIVE_ISOLINES:
      panlib_tess_isoline(&ctx, grid, PANLIB_BARRIER_JM_BARRIER, params,
                          tess_mode);
      break;
   case TESS_PRIMITIVE_TRIANGLES:
      panlib_tess_tri(&ctx, grid, PANLIB_BARRIER_JM_BARRIER, params,
                      tess_mode);
      break;
   case TESS_PRIMITIVE_QUADS:
      panlib_tess_quad(&ctx, grid, PANLIB_BARRIER_JM_BARRIER, params,
                       tess_mode);
      break;
   default:
      UNREACHABLE("invalid tessellation primitive mode");
   }
}

/* On success *out is the indexed indirect TES draw; out->info.indirect is 0
 * when there is nothing to draw. */
static VkResult
launch_tess_v9(struct panvk_cmd_buffer *cmdbuf,
               const struct panvk_draw_data *in_draw,
               struct panvk_draw_data *out_draw)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   struct panvk_cmd_graphics_state *gfx = &cmdbuf->state.gfx;
   const struct panvk_shader *input_vs = gfx->tess.input_vs;
   const struct panvk_shader *tcs = gfx->tess.tcs;
   const struct panvk_shader *tes = gfx->tess.tes;
   const struct vk_dynamic_graphics_state *dyn =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct panvk_draw_info *in = &in_draw->info;

   memset(out_draw, 0, sizeof(*out_draw));

   if (!input_vs || !tcs || !tes || !dev->poly_heap_data)
      return VK_ERROR_FEATURE_NOT_PRESENT;

   /* Indirect draws: the counts are only known on the GPU,
    * panlib_tess_setup_indirect fills them in (Honeykrisp
    * libagx_tess_setup_indirect). PANVK_V9_TESS_NO_INDIRECT=1 skips these
    * draws again (control). */
   const bool indirect = in->indirect.buffer_dev_addr != 0;
   if (indirect && pan_env_flag("PANVK_V9_TESS_NO_INDIRECT"))
      return VK_SUCCESS;

   const uint32_t input_patch_size = dyn->ts.patch_control_points;
   if (!input_patch_size || !tcs->tess.tcs_output_patch_size)
      return VK_ERROR_UNKNOWN;

   const uint32_t patches_per_instance =
      indirect ? 1 : in->vertex.count / input_patch_size;
   const uint32_t patch_count =
      indirect ? 1 : patches_per_instance * in->instance.count;
   if (!patch_count)
      return VK_SUCCESS;

   if (!cmdbuf->cur_batch)
      panvk_per_arch(cmd_open_batch)(cmdbuf);

   VkResult result = panvk_per_arch(cmd_alloc_tls_desc)(cmdbuf, true);
   if (result != VK_SUCCESS)
      return result;

   reset_poly_heap_once_v9(cmdbuf);

   const uint32_t vs_wg_size[3] = {1, 1, 1};
   struct poly_vertex_params vertex_params;
   poly_vertex_params_init(&vertex_params, input_vs->tess.vs_outputs,
                           vs_wg_size);
   if (!indirect)
      poly_vertex_params_set_draw(&vertex_params, in->vertex.count,
                                  in->instance.count);

   uint64_t index_base = 0;
   uint32_t index_range_el = 0;
   if (in->index.index_size) {
      const uint64_t index_offs =
         (uint64_t)in->index.offset * in->index.index_size;
      index_base = in->index.buffer_dev_addr + index_offs;
      index_range_el =
         index_offs < in->index.buffer_size
            ? (in->index.buffer_size - index_offs) / in->index.index_size
            : 0;
      vertex_params.index_buffer = index_base;
      vertex_params.index_size_B = in->index.index_size;
      vertex_params.index_buffer_range_el = index_range_el;
   }

   if (!indirect) {
      const uint64_t vs_output_size = poly_tcs_in_size(
         in->vertex.count * in->instance.count, input_vs->tess.vs_outputs);
      struct pan_ptr vs_output = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc, MAX2(vs_output_size, 16), 16);
      if (!vs_output.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      vertex_params.output_buffer = vs_output.gpu;
   }

   struct pan_ptr vertex_params_ptr = panvk_cmd_upload_dev_mem(
      cmdbuf, desc, &vertex_params, sizeof(vertex_params), 8);
   if (!vertex_params_ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct panvk_tess_info info =
      panvk_tess_info_merge(tcs->tess.info, tes->tess.info);
   enum poly_tess_partitioning partitioning =
      info.spacing == TESS_SPACING_EQUAL
         ? POLY_TESS_PARTITIONING_INTEGER
      : info.spacing == TESS_SPACING_FRACTIONAL_ODD
         ? POLY_TESS_PARTITIONING_FRACTIONAL_ODD
         : POLY_TESS_PARTITIONING_FRACTIONAL_EVEN;

   struct poly_tess_params tess_params = {
      .heap = dev->poly_heap->addr.dev,
      .patch_coord_buffer = dev->poly_heap_data->addr.dev,
      .tcs_stride_el = tcs->tess.tcs_output_stride / 4,
      .input_patch_size = input_patch_size,
      .output_patch_size = tcs->tess.tcs_output_patch_size,
      .tcs_patch_constants = tcs->tess.tcs_nr_patch_outputs,
      .tcs_per_vertex_outputs = tcs->tess.tcs_per_vertex_outputs,
      .patches_per_instance = patches_per_instance,
      .nr_patches = patch_count,
      .partitioning = partitioning,
      .points_mode = info.points,
      .isolines = info.mode == TESS_PRIMITIVE_ISOLINES,
      /* Pipeline statistics: libpoly counts the domain points (32-bit, the
       * low half of the 64-bit counter). */
      .statistic = panvk_v9_stats_counter(cmdbuf, 9),
   };

   if (!tess_params.points_mode && info.mode != TESS_PRIMITIVE_ISOLINES) {
      tess_params.ccw = info.ccw;
      tess_params.ccw ^=
         dyn->ts.domain_origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
   }

   /* Indirect: only the output draw here, the per-patch buffers come from
    * the heap (panlib_tess_setup_indirect). */
   const uint64_t storage_patches = indirect ? 0 : patch_count;
   uint64_t alloc = 0;
   const uint64_t tcs_out_offs = alloc;
   alloc += storage_patches * tess_params.tcs_stride_el * 4;
   const uint64_t coord_allocs_offs = alloc;
   alloc += storage_patches * sizeof(uint32_t);
   const uint64_t counts_offs = alloc;
   alloc += storage_patches * sizeof(uint32_t);
   const uint64_t draw_offs = alloc;
   alloc += sizeof(VkDrawIndexedIndirectCommand);

   struct pan_ptr tess_storage =
      panvk_cmd_alloc_dev_mem(cmdbuf, desc, alloc, 16);
   if (!tess_storage.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   if (indirect)
      memset(tess_storage.cpu, 0, alloc);

   tess_params.tcs_buffer = tess_storage.gpu + tcs_out_offs;
   tess_params.coord_allocs = tess_storage.gpu + coord_allocs_offs;
   tess_params.counts = tess_storage.gpu + counts_offs;
   tess_params.out_draws = tess_storage.gpu + draw_offs;

   struct pan_ptr tess_params_ptr = panvk_cmd_upload_dev_mem(
      cmdbuf, desc, &tess_params, sizeof(tess_params), 8);
   if (!tess_params_ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   gfx->sysvals.common.vertex_param_buffer_poly = vertex_params_ptr.gpu;
   gfx->sysvals.common.tess_param_buffer_poly = tess_params_ptr.gpu;
   panvk_per_arch(cmd_prepare_draw_sysvals)(cmdbuf, in, NULL);

   result = prepare_tess_descs_v9(cmdbuf);
   if (result != VK_SUCCESS)
      return result;

   struct pan_ptr push;
   result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(
      cmdbuf, panvk_shader_sw_variant(input_vs), &push, 1);
   if (result != VK_SUCCESS)
      return result;
   gfx->tess.vs_push_uniforms = push.gpu;

   result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(
      cmdbuf, panvk_shader_only_variant(tcs), &push, 1);
   if (result != VK_SUCCESS)
      return result;
   gfx->tess.tcs_push_uniforms = push.gpu;

   struct pan_ptr jobs[PANLIB_TESS_JOBS];
   const struct panvk_shader_variant *vs_sw =
      panvk_shader_sw_variant(input_vs);
   result = prepare_gfx_cs_job_v9(cmdbuf, vs_sw, &gfx->tess.vs_desc,
                                  gfx->tess.vs_push_uniforms,
                                  indirect ? 1 : in->vertex.count,
                                  indirect ? 1 : in->instance.count, 1,
                                  indirect, &jobs[PANLIB_TESS_JOB_VS]);
   if (result != VK_SUCCESS)
      return result;

   result = prepare_gfx_cs_job_v9(cmdbuf, panvk_shader_only_variant(tcs),
                                  &gfx->tess.tcs_desc,
                                  gfx->tess.tcs_push_uniforms,
                                  patches_per_instance,
                                  indirect ? 1 : in->instance.count, 1,
                                  indirect, &jobs[PANLIB_TESS_JOB_TCS]);
   if (result != VK_SUCCESS)
      return result;

   if (indirect) {
      struct pan_ptr args = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc, sizeof(struct panlib_tess_indirect_params), 16);
      if (!args.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      struct panlib_tess_indirect_params *a = args.cpu;
      memset(a, 0, sizeof(*a));
      a->draw = in->indirect.buffer_dev_addr;
      a->draw_count = in->indirect.count_buffer_dev_addr;
      a->draw_index = in->draw_id;
      a->index_buffer = index_base;
      a->index_size_B = in->index.index_size;
      a->index_buffer_range_el = index_range_el;
      a->tess = tess_params_ptr.gpu;
      a->vp = vertex_params_ptr.gpu;
      a->vs_outputs = input_vs->tess.vs_outputs;
      for (unsigned j = 0; j < PANLIB_TESS_JOBS; j++)
         a->jobs[j] = jobs[j].gpu;

      /* gl_BaseVertex / gl_BaseInstance of the vertex shader. */
      if (jobs[PANLIB_TESS_JOB_VS].gpu) {
         if (shader_uses_sysval(vs_sw, graphics, vs.first_vertex))
            a->first_vertex_sysval =
               gfx->tess.vs_push_uniforms +
               shader_remapped_sysval_offset(
                  vs_sw, sysval_offset(graphics, vs.first_vertex));
         if (shader_uses_sysval(vs_sw, graphics, vs.base_instance))
            a->base_instance_sysval =
               gfx->tess.vs_push_uniforms +
               shader_remapped_sysval_offset(
                  vs_sw, sysval_offset(graphics, vs.base_instance));
      }

      struct panvk_precomp_ctx precomp_ctx =
         panvk_per_arch(precomp_cs)(cmdbuf);
      panlib_tess_setup_indirect(
         &precomp_ctx, panlib_1d(1),
         (enum panlib_barrier)(PANLIB_BARRIER_JM_BARRIER |
                               PANLIB_BARRIER_JM_SUPPRESS_PREFETCH),
         args.gpu);
   }

   queue_gfx_cs_job_v9(cmdbuf, &jobs[PANLIB_TESS_JOB_VS], indirect);
   queue_gfx_cs_job_v9(cmdbuf, &jobs[PANLIB_TESS_JOB_TCS], indirect);

   /* Indirect: fixed grid, the tessellator kernels loop over the patches
    * (PANVK_V9_TESS_INDIRECT_WG workgroups of 64, default 16). */
   uint32_t tess_threads = patch_count;
   if (indirect) {
      const char *wg = getenv("PANVK_V9_TESS_INDIRECT_WG");
      tess_threads = 64 * (wg ? MAX2(atoi(wg), 1) : 16);
   }

   launch_tessellator_v9(cmdbuf, tess_threads, info.mode, tess_params_ptr.gpu,
                         POLY_TESS_MODE_COUNT);

   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   panlib_prefix_sum_tess(&precomp_ctx, panlib_1d(1),
                          PANLIB_BARRIER_JM_BARRIER, tess_params_ptr.gpu);

   launch_tessellator_v9(cmdbuf, tess_threads, info.mode, tess_params_ptr.gpu,
                         POLY_TESS_MODE_WITH_COUNTS);

   gfx->tess.info = info;
   gfx->tess.prim = info.points ? MESA_PRIM_POINTS
                    : info.mode == TESS_PRIMITIVE_ISOLINES
                       ? MESA_PRIM_LINES
                       : MESA_PRIM_TRIANGLES;

   /* The TES push uniforms and descriptors are rebuilt by the draw. */
   gfx_state_set_dirty(cmdbuf, VS);
   gfx_state_set_dirty(cmdbuf, VS_PUSH_UNIFORMS);
   gfx_state_set_dirty(cmdbuf, DESC_STATE);

   *out_draw = (struct panvk_draw_data){
      .info = {
         .index = {
            .buffer_dev_addr = dev->poly_heap_data->addr.dev,
            .buffer_size = pan_kmod_bo_size(dev->poly_heap_data->bo),
            .index_size = 4,
         },
         .vertex.count = 1,
         .instance.count = 1,
         .indirect = {
            .buffer_dev_addr = tess_params.out_draws,
            .draw_count = 1,
            .stride = sizeof(VkDrawIndexedIndirectCommand),
         },
         .prim = gfx->tess.prim,
         .layer_id = in->layer_id,
         .draw_id = in->draw_id,
      },
      .vertex_range = 1,
   };

   return VK_SUCCESS;
}

/* PATCH v9: descriptors of the compute jobs before a GS draw: the API vertex
 * shader (SW variant, vertex driver set layout) and the GS compute variants
 * (compute driver set layout: dummy sampler, then dynamic buffers). */
static VkResult
prepare_gs_descs_v9(struct panvk_cmd_buffer *cmdbuf,
                    const struct panvk_shader *input_vs, uint32_t variants)
{
   struct panvk_cmd_graphics_state *gfx = &cmdbuf->state.gfx;
   const struct panvk_shader *gs = gfx->gs.shader;
   VkResult result;

   uint32_t used_sets = input_vs->desc_info.used_set_mask;
   u_foreach_bit(v, variants)
      used_sets |= gs->gs.cs_desc_info[v].used_set_mask;

   result = panvk_per_arch(cmd_prepare_push_descs)(cmdbuf, &gfx->desc_state,
                                                   used_sets);
   if (result != VK_SUCCESS)
      return result;

   result = prepare_vs_driver_set(cmdbuf, input_vs, &gfx->tess.vs_desc, 1);
   if (result != VK_SUCCESS)
      return result;

   result = panvk_per_arch(cmd_prepare_shader_res_table)(
      cmdbuf, &gfx->desc_state, &input_vs->desc_info, &gfx->tess.vs_desc, 1);
   if (result != VK_SUCCESS)
      return result;

   /* Each GS compute variant has its own descriptor info: compute driver
    * set layout (dummy sampler, then its dynamic buffers) and res table. */
   u_foreach_bit(v, variants) {
      const struct panvk_shader_desc_info *desc_info =
         &gs->gs.cs_desc_info[v];
      struct panvk_shader_desc_state *cs_desc = &gfx->gs.cs_desc[v];
      const uint32_t desc_count = desc_info->dyn_bufs.count + 1;
      struct pan_ptr driver_set = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc, desc_count * PANVK_DESCRIPTOR_SIZE,
         PANVK_DESCRIPTOR_SIZE);
      if (!driver_set.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      struct panvk_opaque_desc *descs = driver_set.cpu;
      pan_cast_and_pack(&descs[0], SAMPLER, cfg) {
         cfg.clamp_integer_array_indices = false;
      }
      panvk_per_arch(cmd_fill_dyn_bufs)(
         &gfx->desc_state, desc_info, (struct mali_buffer_packed *)&descs[1]);
      cs_desc->driver_set.dev_addr = driver_set.gpu;
      cs_desc->driver_set.size = desc_count * PANVK_DESCRIPTOR_SIZE;

      result = panvk_per_arch(cmd_prepare_shader_res_table)(
         cmdbuf, &gfx->desc_state, desc_info, cs_desc, 1);
      if (result != VK_SUCCESS)
         return result;
   }

   return VK_SUCCESS;
}

/* Scratch after struct poly_geometry_params: pipeline-statistics counters
 * (see load_stat_query_address_poly in panvk_vX_shader.c), then a zeroed
 * area for the XFB pointers of a GS without transform feedback. */
#define PANVK_V9_GS_STAT_SINK 128
#define PANVK_V9_GS_SINK_SIZE 256

/* PATCH v9: pipeline statistics of one draw (panlib_stats_add). api is the
 * draw as recorded, tess / gsd the draws the tessellator and the GS wrote. */
static void
emit_draw_stats_v9(struct panvk_cmd_buffer *cmdbuf,
                   const struct panvk_draw_data *api,
                   const struct panvk_draw_data *tess,
                   const struct panvk_draw_data *gsd)
{
   struct pan_ptr args = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, sizeof(struct panlib_stats_params), 16);
   if (!args.gpu)
      return;

   struct panlib_stats_params *a = args.cpu;
   memset(a, 0, sizeof(*a));
   for (unsigned i = 0; i < PANLIB_STATS; i++)
      a->counter[i] = panvk_v9_stats_counter(cmdbuf, i);
   /* Fragment shader invocations: occlusion counter. TES: libpoly. */
   a->counter[7] = 0;
   a->counter[9] = 0;

   const struct panvk_draw_info *in = &api->info;
   if (in->indirect.buffer_dev_addr) {
      a->cmd = in->indirect.buffer_dev_addr;
      a->draw_count = in->indirect.count_buffer_dev_addr;
      a->draw_index = in->draw_id;
   } else {
      a->vertex_count = in->vertex.count;
      a->instance_count = in->instance.count;
   }
   a->prim = in->prim;

   if (tess) {
      a->patch_size = cmdbuf->vk.dynamic_graphics_state.ts.patch_control_points;
      a->tess_draw = tess->info.indirect.buffer_dev_addr;
      a->tess_verts_per_prim = tess->info.prim == MESA_PRIM_POINTS ? 1
                               : tess->info.prim == MESA_PRIM_LINES ? 2
                                                                     : 3;
   }

   if (gsd) {
      a->gs_invocations = MAX2(cmdbuf->state.gfx.gs.shader->gs.invocations, 1);
      if (gsd->info.indirect.buffer_dev_addr) {
         a->gs_draw = gsd->info.indirect.buffer_dev_addr;
      } else {
         a->gs_count = gsd->info.vertex.count;
         a->gs_instances = gsd->info.instance.count;
      }
   }

   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   panlib_stats_add(&precomp_ctx, panlib_1d(1),
                    (enum panlib_barrier)(PANLIB_BARRIER_JM_BARRIER |
                                          PANLIB_BARRIER_JM_SUPPRESS_PREFETCH),
                    args.gpu);
}

/* PATCH v9: an indexed strip (or fan) with primitive restart in front of a
 * geometry shader is rewritten as a list without restarts by
 * panlib_unroll_restart (geometry.cl), and the GS reads that indexed
 * indirect draw. PANVK_V9_GS_NO_UNROLL=1 skips it (control). */
static bool
unroll_restart_v9(struct panvk_cmd_buffer *cmdbuf,
                  const struct panvk_draw_data *in_draw,
                  struct panvk_draw_data *out_draw)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const struct panvk_draw_info *in = &in_draw->info;

   if (!in->index.index_size || !in->index.restart_enable ||
       in->prim == u_decomposed_prim(in->prim) || !dev->poly_heap_data ||
       pan_env_flag("PANVK_V9_GS_NO_UNROLL"))
      return false;

   const bool indirect = in->indirect.buffer_dev_addr != 0;
   if (!indirect && (!in->vertex.count || !in->instance.count))
      return false;

   if (!cmdbuf->cur_batch)
      panvk_per_arch(cmd_open_batch)(cmdbuf);
   reset_poly_heap_once_v9(cmdbuf);

   uint64_t in_cmd = in->indirect.buffer_dev_addr;
   if (!indirect) {
      const VkDrawIndexedIndirectCommand cmd = {
         .indexCount = in->vertex.count,
         .instanceCount = in->instance.count,
         .firstIndex = in->index.offset,
         .vertexOffset = in->vertex.base,
         .firstInstance = in->instance.base,
      };
      struct pan_ptr p =
         panvk_cmd_upload_dev_mem(cmdbuf, desc, &cmd, sizeof(cmd), 16);
      if (!p.gpu)
         return false;
      in_cmd = p.gpu;
   }

   struct pan_ptr out_cmd = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, sizeof(VkDrawIndexedIndirectCommand), 16);
   struct pan_ptr args = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, sizeof(struct panlib_unroll_restart_params), 16);
   if (!out_cmd.gpu || !args.gpu)
      return false;

   const uint32_t isz = in->index.index_size;
   struct panlib_unroll_restart_params *a = args.cpu;
   memset(a, 0, sizeof(*a));
   a->heap = dev->poly_heap->addr.dev;
   a->index_buffer = in->index.buffer_dev_addr;
   a->index_buffer_range_el = in->index.buffer_size / isz;
   a->in_draw = in_cmd;
   a->out_draw = out_cmd.gpu;
   a->draw_count = indirect ? in->indirect.count_buffer_dev_addr : 0;
   a->draw_index = in->draw_id;
   a->index_size_B = isz;
   a->restart_index = isz == 1 ? 0xff : isz == 2 ? 0xffff : 0xffffffff;
   a->flatshade_first =
      cmdbuf->vk.dynamic_graphics_state.rs.provoking_vertex ==
      VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT;
   a->prim = in->prim;

   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   panlib_unroll_restart(
      &precomp_ctx, panlib_1d(1),
      (enum panlib_barrier)(PANLIB_BARRIER_JM_BARRIER |
                            PANLIB_BARRIER_JM_SUPPRESS_PREFETCH),
      args.gpu);

   *out_draw = *in_draw;
   out_draw->info.index.buffer_dev_addr = dev->poly_heap_data->addr.dev;
   out_draw->info.index.buffer_size = pan_kmod_bo_size(dev->poly_heap_data->bo);
   out_draw->info.index.offset = 0;
   out_draw->info.index.restart_enable = false;
   out_draw->info.vertex.base = 0;
   out_draw->info.vertex.count = 1;
   out_draw->info.instance.count = 1;
   out_draw->info.indirect.buffer_dev_addr = out_cmd.gpu;
   out_draw->info.indirect.count_buffer_dev_addr = 0;
   out_draw->info.indirect.draw_count = 1;
   out_draw->info.indirect.stride = 0;
   out_draw->info.prim = u_decomposed_prim(in->prim);
   return true;
}

/* PATCH v9: geometry shader through libpoly, port of Honeykrisp
 * hk_launch_gs_prerast() for direct draws: the API vertex shader (SW
 * variant) writes its outputs to memory, the GS MAIN variant runs once per
 * input primitive and instance and writes the output index buffer, then the
 * draw runs the GS RAST variant as the hardware vertex shader. Every job
 * carries the JM job barrier. Indirect draws, GS after tessellation and XFB
 * from the GS were added on top of it later.
 * 🧪 EXPERIMENTAL. On success *out is the rasterization draw;
 * out->info.vertex.count is 0 when there is nothing to draw. */
static VkResult
launch_gs_v9(struct panvk_cmd_buffer *cmdbuf,
             const struct panvk_draw_data *in_draw,
             struct panvk_draw_data *out_draw)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   struct panvk_cmd_graphics_state *gfx = &cmdbuf->state.gfx;
   const struct panvk_shader *gs = gfx->gs.shader;
   const struct vk_dynamic_graphics_state *dyn =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct panvk_draw_info *in = &in_draw->info;

   memset(out_draw, 0, sizeof(*out_draw));

   /* The stage before the GS runs as compute (its SW variant) and writes the
    * vertices the GS reads: the API vertex shader, or the TES after
    * tessellation. In the TES case in_draw is the indexed indirect draw
    * written by the tessellator. */
   const struct panvk_shader *tes = gfx->tess.tes;
   const struct panvk_shader *input_vs = tes ? tes : gfx->tess.input_vs;
   if (!input_vs || !gs)
      return VK_ERROR_FEATURE_NOT_PRESENT;

   const struct panvk_shader_variant *input_sw =
      panvk_shader_sw_variant(input_vs);

   /* Vertex and instance counts only known on the GPU:
    * panlib_gs_setup_indirect fills the parameters and the job grids.
    * PANVK_V9_GS_NO_INDIRECT=1 skips these draws again (control). */
   const bool indirect = in->indirect.buffer_dev_addr != 0;
   if (indirect && (pan_env_flag("PANVK_V9_GS_NO_INDIRECT") || !dev->poly_heap_data))
      return VK_SUCCESS;

   const struct poly_gs_info *gsi = &gs->gs.info;

   /* Transform feedback from the GS: COUNT (dynamic vertex counts), the
    * prefix sum, PRE (XFB buffer setup and offsets) and the XFB copy of
    * RAST run as well. PANVK_V9_GS_NO_XFB=1 skips them (control). */
   const bool xfb = gsi->xfb && gfx->xfb.enabled && gfx->xfb.offsets.gpu &&
                    !pan_env_flag("PANVK_V9_GS_NO_XFB");
   const enum mesa_prim in_prim = in->prim;

   if (in->index.index_size && in->index.restart_enable && !tes &&
       in_prim != u_decomposed_prim(in_prim))
      mesa_logw_once("panvk: v9: primitive restart before a geometry shader "
                     "is not unrolled (PANVK_V9_GS_NO_UNROLL)");

   const uint32_t prims =
      indirect ? 1 : u_decomposed_prims_for_vertices(in_prim, in->vertex.count);
   if (!indirect && (!prims || !in->instance.count))
      return VK_SUCCESS;

   if (!cmdbuf->cur_batch)
      panvk_per_arch(cmd_open_batch)(cmdbuf);

   VkResult result = panvk_per_arch(cmd_alloc_tls_desc)(cmdbuf, true);
   if (result != VK_SUCCESS)
      return result;

   if (indirect)
      reset_poly_heap_once_v9(cmdbuf);

   /* Vertex shader outputs, written by the SW VS, read by MAIN and RAST. */
   const uint32_t one[3] = {1, 1, 1};
   struct poly_vertex_params vertex_params;
   poly_vertex_params_init(&vertex_params, input_vs->tess.vs_outputs, one);

   uint64_t index_base = 0;
   uint32_t index_range_el = 0;
   if (in->index.index_size) {
      const uint64_t index_offs =
         (uint64_t)in->index.offset * in->index.index_size;
      index_base = in->index.buffer_dev_addr + index_offs;
      index_range_el =
         index_offs < in->index.buffer_size
            ? (in->index.buffer_size - index_offs) / in->index.index_size
            : 0;
      vertex_params.index_buffer = index_base;
      vertex_params.index_size_B = in->index.index_size;
      vertex_params.index_buffer_range_el = index_range_el;
   }

   if (!indirect) {
      poly_vertex_params_set_draw(&vertex_params, in->vertex.count,
                                  in->instance.count);

      const uint64_t vs_output_size = poly_tcs_in_size(
         in->vertex.count * in->instance.count, input_vs->tess.vs_outputs);
      struct pan_ptr vs_output = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc, MAX2(vs_output_size, 16), 16);
      if (!vs_output.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      vertex_params.output_buffer = vs_output.gpu;
   }

   struct pan_ptr vertex_params_ptr = panvk_cmd_upload_dev_mem(
      cmdbuf, desc, &vertex_params, sizeof(vertex_params), 8);
   if (!vertex_params_ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   /* Geometry parameters plus the scratch sink. */
   struct pan_ptr gp_ptr = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc,
      sizeof(struct poly_geometry_params) + PANVK_V9_GS_SINK_SIZE, 16);
   if (!gp_ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   memset(gp_ptr.cpu, 0,
          sizeof(struct poly_geometry_params) + PANVK_V9_GS_SINK_SIZE);
   const uint64_t sink =
      gp_ptr.gpu + sizeof(struct poly_geometry_params) + PANVK_V9_GS_STAT_SINK;

   struct poly_geometry_params params;
   poly_geometry_params_init(&params, in_prim, one);
   params.flat_outputs = 0;
   for (unsigned i = 0; i < POLY_MAX_SO_BUFFERS; ++i) {
      params.xfb_offs_ptrs[i] = sink;
      params.xfb_base_original[i] = sink;
      params.xfb_size[i] = 0;
   }
   if (xfb) {
      for (unsigned i = 0; i < MIN2(POLY_MAX_SO_BUFFERS,
                                    PANVK_MAX_XFB_BUFFERS); ++i) {
         params.xfb_offs_ptrs[i] =
            gfx->xfb.offsets.gpu + i * sizeof(uint32_t);
         if (gfx->xfb.buffers[i].address) {
            params.xfb_base_original[i] = gfx->xfb.buffers[i].address;
            params.xfb_size[i] = gfx->xfb.buffers[i].size;
         }
      }
   }
   for (unsigned i = 0; i < POLY_MAX_VERTEX_STREAMS; ++i) {
      params.xfb_prims_generated_counter[i] = sink + 32;
      params.prims_generated_counter[i] = sink + 40;
      params.xfb_overflow[i] = sink + 48;
   }
   params.xfb_any_overflow = sink + 56;
   params.count_buffer_stride = gsi->count_words * 4;

   if (!indirect)
      poly_geometry_params_set_draw(&params, in_prim, gsi->shape,
                                    gsi->max_indices, in->vertex.count,
                                    in->instance.count);

   /* With a prefix sum the indirect setup allocates the count buffer from
    * the heap. */
   if (params.count_buffer_stride && !(indirect && gsi->prefix_sum)) {
      const uint64_t size =
         gsi->prefix_sum
            ? (uint64_t)params.input_primitives * params.count_buffer_stride
            : 16;
      struct pan_ptr cb = panvk_cmd_alloc_dev_mem(cmdbuf, desc, size, 16);
      if (!cb.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      memset(cb.cpu, 0, size);
      params.count_buffer = cb.gpu;
   }

   const uint32_t index_count = indirect ? 1 : params.draw.index_count;
   const uint32_t instance_count = indirect ? 1 : params.draw.instance_count;
   if (!index_count || !instance_count)
      return VK_SUCCESS;

   uint64_t ib = 0, ib_size = 0;
   uint32_t index_size = poly_gs_index_size(gsi->shape);
   if (gsi->shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
      if (indirect) {
         /* Allocated from the heap by the setup, p->draw.first_index is
          * relative to the heap base. */
         ib = dev->poly_heap_data->addr.dev;
         ib_size = pan_kmod_bo_size(dev->poly_heap_data->bo);
      } else {
         ib_size = (uint64_t)index_count * 4;
         struct pan_ptr p = panvk_cmd_alloc_dev_mem(cmdbuf, desc, ib_size, 64);
         if (!p.gpu)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
         ib = p.gpu;
         params.output_index_buffer = ib;
      }
   } else if (gsi->shape == POLY_GS_SHAPE_STATIC_INDEXED) {
      ib_size = sizeof(gsi->topology);
      struct pan_ptr p = panvk_cmd_upload_dev_mem(
         cmdbuf, desc, gsi->topology, sizeof(gsi->topology), 64);
      if (!p.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      ib = p.gpu;
      /* Only read by the XFB variant, MAIN writes no index buffer here. */
      params.output_index_buffer = ib;
   }

   memcpy(gp_ptr.cpu, &params, sizeof(params));

   gfx->sysvals.common.vertex_param_buffer_poly = vertex_params_ptr.gpu;
   gfx->sysvals.common.geometry_param_buffer_poly = gp_ptr.gpu;
   gfx->sysvals.common.provoking_last_poly =
      dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT;
   gfx->sysvals.common.rasterization_stream_poly =
      dyn->rs.rasterization_stream;
   panvk_per_arch(cmd_prepare_draw_sysvals)(cmdbuf, in, NULL);

   uint32_t variants = BITFIELD_BIT(PANVK_GS_VARIANT_MAIN);
   if (xfb) {
      variants |= BITFIELD_BIT(PANVK_GS_VARIANT_PRE) |
                  BITFIELD_BIT(PANVK_GS_VARIANT_XFB);
      if (gsi->count_words)
         variants |= BITFIELD_BIT(PANVK_GS_VARIANT_COUNT);
   }
   u_foreach_bit(v, variants) {
      if (!panvk_shader_gs_variant(gs, v)->bin_size)
         variants &= ~BITFIELD_BIT(v);
   }

   result = prepare_gs_descs_v9(cmdbuf, input_vs, variants);
   if (result != VK_SUCCESS)
      return result;

   struct pan_ptr push;
   result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(cmdbuf, input_sw,
                                                         &push, 1);
   if (result != VK_SUCCESS)
      return result;
   gfx->tess.vs_push_uniforms = push.gpu;

   u_foreach_bit(v, variants) {
      result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(
         cmdbuf, panvk_shader_gs_variant(gs, v), &push, 1);
      if (result != VK_SUCCESS)
         return result;
      gfx->gs.cs_push_uniforms[v] = push.gpu;
   }

   /* Build the jobs first so an indirect setup can patch their grids, then
    * queue them in order behind it. */
   struct pan_ptr jobs[PANLIB_GS_JOBS] = {0};
   static const enum panvk_gs_variant job_variant[PANLIB_GS_JOBS] = {
      [PANLIB_GS_JOB_COUNT] = PANVK_GS_VARIANT_COUNT,
      [PANLIB_GS_JOB_MAIN] = PANVK_GS_VARIANT_MAIN,
      [PANLIB_GS_JOB_XFB] = PANVK_GS_VARIANT_XFB,
   };

   result = prepare_gfx_cs_job_v9(cmdbuf, input_sw, &gfx->tess.vs_desc,
                                  gfx->tess.vs_push_uniforms,
                                  indirect ? 1 : in->vertex.count,
                                  indirect ? 1 : in->instance.count, 1,
                                  indirect, &jobs[PANLIB_GS_JOB_VS]);
   if (result != VK_SUCCESS)
      return result;

   for (unsigned j = PANLIB_GS_JOB_COUNT; j < PANLIB_GS_JOBS; j++) {
      const enum panvk_gs_variant v = job_variant[j];
      if (!(variants & BITFIELD_BIT(v)))
         continue;

      const uint32_t x = indirect                 ? 1
                         : j == PANLIB_GS_JOB_XFB ? index_count
                                                  : prims;
      const uint32_t y = indirect                 ? 1
                         : j == PANLIB_GS_JOB_XFB ? instance_count
                                                  : in->instance.count;
      result = prepare_gfx_cs_job_v9(cmdbuf, panvk_shader_gs_variant(gs, v),
                                     &gfx->gs.cs_desc[v],
                                     gfx->gs.cs_push_uniforms[v], x, y, 1,
                                     indirect, &jobs[j]);
      if (result != VK_SUCCESS)
         return result;
   }

   if (indirect) {
      struct pan_ptr args = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc, sizeof(struct panlib_gs_indirect_params), 16);
      if (!args.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      struct panlib_gs_indirect_params *a = args.cpu;
      memset(a, 0, sizeof(*a));
      a->draw = in->indirect.buffer_dev_addr;
      a->draw_count = in->indirect.count_buffer_dev_addr;
      a->draw_index = in->draw_id;
      a->index_buffer = index_base;
      a->index_size_B = in->index.index_size;
      a->index_buffer_range_el = index_range_el;
      a->vp = vertex_params_ptr.gpu;
      a->p = gp_ptr.gpu;
      a->heap = dev->poly_heap->addr.dev;
      a->vs_outputs = input_vs->tess.vs_outputs;
      a->prim = in_prim;
      a->prefix_sum = xfb && gsi->prefix_sum;
      a->max_indices = gsi->max_indices;
      a->shape = gsi->shape;
      for (unsigned j = 0; j < PANLIB_GS_JOBS; j++)
         a->jobs[j] = jobs[j].gpu;

      /* gl_BaseVertex / gl_BaseInstance of the API vertex shader. */
      if (!tes && jobs[PANLIB_GS_JOB_VS].gpu) {
         if (shader_uses_sysval(input_sw, graphics, vs.first_vertex))
            a->first_vertex_sysval =
               gfx->tess.vs_push_uniforms +
               shader_remapped_sysval_offset(
                  input_sw, sysval_offset(graphics, vs.first_vertex));
         if (shader_uses_sysval(input_sw, graphics, vs.base_instance))
            a->base_instance_sysval =
               gfx->tess.vs_push_uniforms +
               shader_remapped_sysval_offset(
                  input_sw, sysval_offset(graphics, vs.base_instance));
      }

      struct panvk_precomp_ctx precomp_ctx =
         panvk_per_arch(precomp_cs)(cmdbuf);
      panlib_gs_setup_indirect(
         &precomp_ctx, panlib_1d(1),
         (enum panlib_barrier)(PANLIB_BARRIER_JM_BARRIER |
                               PANLIB_BARRIER_JM_SUPPRESS_PREFETCH),
         args.gpu);
   }

   queue_gfx_cs_job_v9(cmdbuf, &jobs[PANLIB_GS_JOB_VS], indirect);
   queue_gfx_cs_job_v9(cmdbuf, &jobs[PANLIB_GS_JOB_COUNT], indirect);

   if (xfb && gsi->prefix_sum) {
      struct panvk_precomp_ctx precomp_ctx =
         panvk_per_arch(precomp_cs)(cmdbuf);
      panlib_prefix_sum_geom(&precomp_ctx, panlib_1d(1),
                             PANLIB_BARRIER_JM_BARRIER, gp_ptr.gpu);
   }

   if (variants & BITFIELD_BIT(PANVK_GS_VARIANT_PRE)) {
      result = launch_gfx_cs_v9(
         cmdbuf, panvk_shader_gs_variant(gs, PANVK_GS_VARIANT_PRE),
         &gfx->gs.cs_desc[PANVK_GS_VARIANT_PRE],
         gfx->gs.cs_push_uniforms[PANVK_GS_VARIANT_PRE], 1, 1, 1);
      if (result != VK_SUCCESS)
         return result;
   }

   queue_gfx_cs_job_v9(cmdbuf, &jobs[PANLIB_GS_JOB_MAIN], indirect);
   queue_gfx_cs_job_v9(cmdbuf, &jobs[PANLIB_GS_JOB_XFB], indirect);

   /* The RAST push uniforms and descriptors are rebuilt by the draw. */
   gfx_state_set_dirty(cmdbuf, VS);
   gfx_state_set_dirty(cmdbuf, VS_PUSH_UNIFORMS);
   gfx_state_set_dirty(cmdbuf, DESC_STATE);

   *out_draw = (struct panvk_draw_data){
      .info = {
         .index = {
            .buffer_dev_addr = ib,
            .buffer_size = ib_size,
            .index_size = ib ? index_size : 0,
            .restart_enable = ib != 0,
         },
         .vertex.count = index_count,
         .instance.count = instance_count,
         .prim = gsi->mode,
         .layer_id = in->layer_id,
         .draw_id = in->draw_id,
      },
      .vertex_range = index_count,
   };

   /* The rasterization draw reads the counts the setup wrote. */
   if (indirect) {
      out_draw->info.indirect.buffer_dev_addr =
         gp_ptr.gpu + offsetof(struct poly_geometry_params, draw);
      out_draw->info.indirect.draw_count = 1;
      out_draw->info.indirect.stride = 0;
   }

   return VK_SUCCESS;
}

/* PATCH v9: a vertex shader whose only outputs are XFB captures (no
 * gl_Position, no point size) compiles to an empty hardware variant, so the
 * draw has nothing to rasterize and is skipped. The capture must still run:
 * prepare only what the XFB compute job reads (TLS, sysvals, push and VS
 * descriptors), then launch it. */
static void
launch_xfb_only_v9(struct panvk_cmd_buffer *cmdbuf,
                   struct panvk_draw_data *draw)
{
   struct panvk_cmd_graphics_state *gfx = &cmdbuf->state.gfx;

   if (!gfx->xfb.enabled || !gfx->vs.shader ||
       gfx->vs.shader->vk.stage != MESA_SHADER_VERTEX ||
       !panvk_shader_xfb_variant(gfx->vs.shader))
      return;

   if (!cmdbuf->cur_batch)
      panvk_per_arch(cmd_open_batch)(cmdbuf);

   struct panvk_batch *batch = cmdbuf->cur_batch;
   if (panvk_per_arch(cmd_alloc_tls_desc)(cmdbuf, true) != VK_SUCCESS)
      return;
   draw->tls = batch->tls.gpu;

   panvk_per_arch(cmd_prepare_draw_sysvals)(cmdbuf, &draw->info, NULL);

   if (panvk_per_arch(cmd_prepare_push_descs)(
          cmdbuf, &gfx->desc_state, gfx->vs.shader->desc_info.used_set_mask) !=
       VK_SUCCESS)
      return;

   if (prepare_vs_driver_set(cmdbuf, gfx->vs.shader, &gfx->vs.desc, 1) !=
       VK_SUCCESS)
      return;

   /* The next real draw must rebuild its VS descriptors. */
   gfx_state_set_dirty(cmdbuf, DESC_STATE);

   launch_xfb_v9(cmdbuf, draw);
}

/* Layered rendering and multiview on v9 JM, same scheme as the Bifrost path
 * (prepare_draw_layer): every draw is recorded once per framebuffer layer (or
 * per view in the view mask) with its own tiler context and layer_id sysval.
 * The fragment jobs are already emitted per layer when the batch is closed.
 * Before this only layer 0 was drawn, so for example a single
 * vkCmdCopyBufferToImage region with layerCount 6 into an AFBC cube (copied
 * through the graphics path) left layers 1..5 unwritten.
 * PANVK_V9_LAYER0_ONLY=1 restores the old behaviour as a negative control. */
static void
panvk_cmd_draw_v9(struct panvk_cmd_buffer *cmdbuf,
                  struct panvk_draw_data *draw)
{
   /* Test hook: PANVK_V9_FORCE_INDIRECT=1 turns a direct draw that goes
    * through tessellation or a GS into the equivalent indirect draw, so the
    * GPU-side setup paths (panlib_tess_setup_indirect,
    * panlib_gs_setup_indirect) run on every tess/GS test case. */
   struct panvk_draw_data forced;
   if (!draw->info.indirect.buffer_dev_addr &&
       (cmdbuf->state.gfx.tess.tes || cmdbuf->state.gfx.gs.shader) &&
       pan_env_flag("PANVK_V9_FORCE_INDIRECT")) {
      const bool indexed = draw->info.index.index_size != 0;
      struct pan_ptr cmd = panvk_cmd_alloc_dev_mem(cmdbuf, desc, 20, 4);
      if (!cmd.gpu)
         return;
      uint32_t *c = cmd.cpu;
      c[0] = draw->info.vertex.count;
      c[1] = draw->info.instance.count;
      if (indexed) {
         c[2] = draw->info.index.offset;
         c[3] = draw->info.vertex.base;
         c[4] = draw->info.instance.base;
      } else {
         c[2] = draw->info.vertex.base;
         c[3] = draw->info.instance.base;
         c[4] = 0;
      }
      forced = *draw;
      forced.info.index.offset = 0;
      forced.info.vertex.base = 0;
      forced.info.instance.base = 0;
      forced.info.vertex.count = 1;
      forced.info.instance.count = 1;
      forced.info.indirect.buffer_dev_addr = cmd.gpu;
      forced.info.indirect.draw_count = 1;
      forced.info.indirect.stride = 0;
      draw = &forced;
   }
   const struct panvk_draw_data *api_draw = draw;
   const struct panvk_draw_data *stats_tess = NULL, *stats_gs = NULL;

   /* PATCH v9: tessellation prepass, then draw the TES output. */
   struct panvk_draw_data tess_draw;
   if (cmdbuf->state.gfx.tess.tes) {
      if (launch_tess_v9(cmdbuf, draw, &tess_draw) != VK_SUCCESS ||
          !tess_draw.info.indirect.buffer_dev_addr)
         return;
      draw = &tess_draw;
      stats_tess = &tess_draw;
   }

   /* PATCH v9: geometry shader prepass, then draw the GS RAST variant. The
    * rasterized topology is the GS output (point list, line strip or
    * triangle strip with restart), not the API one, so the input-assembly
    * dynamic state is swapped for the duration of the draw. */
   struct panvk_draw_data gs_draw;
   struct vk_input_assembly_state *ia = &cmdbuf->vk.dynamic_graphics_state.ia;
   const struct vk_input_assembly_state saved_ia = *ia;
   const bool gs = cmdbuf->state.gfx.gs.shader != NULL;
   /* A tessellated draw rasterizes the tessellator output, not patches:
    * without this, points mode ignored the TES gl_PointSize (1 pixel
    * points). PANVK_V9_TESS_KEEP_IA=1 keeps the API topology (control). */
   const bool tess_ia = !gs && cmdbuf->state.gfx.tess.tes &&
                        !pan_env_flag("PANVK_V9_TESS_KEEP_IA");
   if (gs || tess_ia) {
      if (gs) {
         struct panvk_draw_data unrolled;
         if (!cmdbuf->state.gfx.tess.tes &&
             unroll_restart_v9(cmdbuf, draw, &unrolled))
            draw = &unrolled;
         if (launch_gs_v9(cmdbuf, draw, &gs_draw) != VK_SUCCESS ||
             !gs_draw.info.vertex.count)
            return;
         draw = &gs_draw;
         stats_gs = &gs_draw;
      }
      ia->primitive_topology =
         draw->info.prim == MESA_PRIM_POINTS ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST
         : draw->info.prim == MESA_PRIM_LINE_STRIP
            ? VK_PRIMITIVE_TOPOLOGY_LINE_STRIP
         : draw->info.prim == MESA_PRIM_LINES ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST
         : draw->info.prim == MESA_PRIM_TRIANGLES
            ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
            : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
      ia->primitive_restart_enable = draw->info.index.restart_enable;
      BITSET_SET(cmdbuf->vk.dynamic_graphics_state.dirty,
                 MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY);
      BITSET_SET(cmdbuf->vk.dynamic_graphics_state.dirty,
                 MESA_VK_DYNAMIC_IA_PRIMITIVE_RESTART_ENABLE);
   }

   /* Pipeline statistics of the draw. PANVK_V9_STATS_NO_DRAW=1 skips it,
    * the draw counters then stay 0 (control). */
   if (cmdbuf->state.stats.addr && !pan_env_flag("PANVK_V9_STATS_NO_DRAW"))
      emit_draw_stats_v9(cmdbuf, api_draw, stats_tess, stats_gs);

   uint32_t view_mask = cmdbuf->state.gfx.render.view_mask;
   uint32_t layer_count = view_mask ? util_bitcount(view_mask)
                                    : cmdbuf->state.gfx.render.layer_count;

   /* PATCH v9: without a gl_Layer write the primitives go to layer 0 only
    * (Vulkan default layer), so do not replicate the draw. Multiview is
    * unaffected. PANVK_V9_ALL_LAYERS=1 restores the old replication. */
   const struct panvk_shader *hw_vs = cmdbuf->state.gfx.vs.shader;
   if (!view_mask && hw_vs && !hw_vs->writes_layer &&
       !pan_env_flag("PANVK_V9_ALL_LAYERS"))
      layer_count = 1;

   /* multiViewport: Mali has one viewport per draw. When the last
    * pre-rasterization stage writes gl_ViewportIndex the draw is recorded
    * once per viewport and the shader moves the vertices of the other
    * viewports out of the way (lower_viewport_writes). Otherwise only
    * viewport 0 is used, as the spec says. */
   const uint32_t vp_count =
      hw_vs && hw_vs->writes_viewport
         ? MAX2(cmdbuf->vk.dynamic_graphics_state.vp.viewport_count, 1)
         : 1;

   for (uint32_t v = 0; v < vp_count; v++) {
      draw->info.viewport_id = v;
      uint32_t mask = view_mask;
      if (layer_count <= 1 || pan_env_flag("PANVK_V9_LAYER0_ONLY")) {
         draw->info.layer_id = mask ? u_bit_scan(&mask) : 0;
         panvk_cmd_draw_v9_layer(cmdbuf, draw);
      } else {
         for (uint32_t i = 0; i < layer_count; i++) {
            draw->info.layer_id = mask ? u_bit_scan(&mask) : i;
            panvk_cmd_draw_v9_layer(cmdbuf, draw);
         }
      }
   }
   draw->info.viewport_id = 0;

   if (gs || tess_ia) {
      *ia = saved_ia;
      BITSET_SET(cmdbuf->vk.dynamic_graphics_state.dirty,
                 MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY);
      BITSET_SET(cmdbuf->vk.dynamic_graphics_state.dirty,
                 MESA_VK_DYNAMIC_IA_PRIMITIVE_RESTART_ENABLE);
   }
}

static void
panvk_cmd_draw_v9_layer(struct panvk_cmd_buffer *cmdbuf,
                        struct panvk_draw_data *draw)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   struct panvk_batch *batch = cmdbuf->cur_batch;

   /* If there's no vertex shader, we can skip the draw. On v9 the vertex
    * program is an SPD, not an RSD. */
   /* PATCH v9: sepasang dengan fix di panvk_emit_vs_position_v9 --
    * cek alokasi harus ikut topology juga, bukan selalu pos_triangles,
    * kalau tidak draw titik (MESA_PRIM_POINTS) selalu di-skip diam-diam
    * walau shadernya valid (dialokasikan di slot pos_points, bukan
    * pos_triangles). */
   const bool pos_spd_alloc = draw->info.prim == MESA_PRIM_POINTS
                                 ? panvk_priv_mem_check_alloc(vs->spds.pos_points)
                                 : panvk_priv_mem_check_alloc(vs->spds.pos_triangles);
   if (!pos_spd_alloc) {
      if (draw->info.layer_id == 0)
         launch_xfb_only_v9(cmdbuf, draw);
      return;
   }

   assert(batch);

   /* Needs to be done before get_fs() is called because it depends on
    * fs.required being initialized. */
   cmdbuf->state.gfx.fs.required =
      fs_required(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state);

   if (prepare_draw_v9(cmdbuf, draw) != VK_SUCCESS)
      return;

   /* PATCH v9: prepare_draw_v9() closes the batch and opens a new one when
    * the 16-bit job index would overflow. The pointer read at the top of this
    * function is then the closed batch, and the job was chained into its
    * vertex/tiler chain while using the new batch's tiler context: every
    * draw after the split got lost (CTS
    * api.command_buffers.record_many_draws_secondary_2, 65k+ jobs in one
    * render pass). The Bifrost path reads cur_batch after prepare_draw() for
    * the same reason. PANVK_V9_STALE_BATCH=1 restores the old behaviour
    * (negative control). */
   static int stale_batch = -1;
   if (stale_batch < 0)
      stale_batch = pan_env_flag("PANVK_V9_STALE_BATCH") != NULL;
   if (!stale_batch)
      batch = cmdbuf->cur_batch;

   /* Transform feedback runs once per draw, not once per layer
    * (multiview with transform feedback is not allowed). */
   if (draw->info.layer_id == 0 && launch_xfb_v9(cmdbuf, draw) != VK_SUCCESS)
      return;

   /* Indirect draws: the counts are not known at record time, so the job is
    * built with placeholders, parked as NOT_STARTED so the hardware cannot run
    * it yet, and a helper kernel then overwrites the count fields from the
    * application's buffer and promotes the job header to MALLOC_VERTEX. Same
    * shape as the indirect compute dispatch path in panvk_vX_cmd_dispatch.c.
    *
    * The helper is queued first so its job index can be used as the dependency
    * for the draw. */
   const bool is_indirect = draw->info.indirect.buffer_dev_addr != 0;
   unsigned indirect_dep = 0;

   /* Addresses of the draw-parameter sysvals inside this draw's freshly
    * allocated VS push uniform block, or 0 when the shader does not read them.
    * The indirect helper fills them from the application's buffer, since the
    * CPU only had placeholders. Same scheme as the Bifrost path above. */
   uint64_t fv_sysval = 0, bi_sysval = 0;
   if (is_indirect) {
      /* Must be the variant prepare_draw_v9 built the push uniforms from,
       * otherwise the remapped offsets point at the wrong slots. */
      const struct panvk_shader_variant *ivs =
         panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
      if (shader_uses_sysval(ivs, graphics, vs.first_vertex))
         fv_sysval = cmdbuf->state.gfx.vs.push_uniforms +
                     shader_remapped_sysval_offset(
                        ivs, sysval_offset(graphics, vs.first_vertex));
      if (shader_uses_sysval(ivs, graphics, vs.base_instance))
         bi_sysval = cmdbuf->state.gfx.vs.push_uniforms +
                     shader_remapped_sysval_offset(
                        ivs, sysval_offset(graphics, vs.base_instance));

      /* Negative control: withhold the pointers, reproducing the behaviour
       * before the sysvals were patched. */
      if (pan_env_flag("PANVK_INDIRECT_NO_SYSVAL"))
         fv_sysval = bi_sysval = 0;
   }

   /* Instance-rate attributes whose descriptor Offset the helper must bump by
    * firstInstance * binding stride. Descriptors 0..MAX_VS_ATTRIBS-1 of the VS
    * driver set are the attributes (prepare_vs_driver_set). */
   uint64_t vs_attribs = 0, inst_strides = 0;
   uint32_t inst_mask = 0;
   if (is_indirect && !pan_env_flag("PANVK_INDIRECT_NO_INSTATTR")) {
      const struct vk_dynamic_graphics_state *idyns =
         &cmdbuf->vk.dynamic_graphics_state;
      const struct vk_vertex_input_state *ivi = idyns->vi;
      uint32_t strides[MAX_VS_ATTRIBS] = {0};
      u_foreach_bit(a, ivi->attributes_valid) {
         const uint32_t b = ivi->attributes[a].binding;
         if (ivi->bindings[b].input_rate != VK_VERTEX_INPUT_RATE_INSTANCE)
            continue;
         inst_mask |= BITFIELD_BIT(a);
         strides[a] = idyns->vi_binding_strides[b];
      }
      if (inst_mask) {
         struct pan_ptr sp =
            panvk_cmd_alloc_dev_mem(cmdbuf, desc, sizeof(strides), 4);
         if (!sp.gpu)
            return;
         memcpy(sp.cpu, strides, sizeof(strides));
         inst_strides = sp.gpu;
         vs_attribs = cmdbuf->state.gfx.vs.desc.driver_set.dev_addr;
      }
   }

   /* T4.5.8 experiment. Removing the dependency entirely (PANVK_INDIRECT_NODEP)
    * changed nothing observable in 40 runs, which means the ordering could have
    * been coming from chain-order serialisation rather than from the dependency.
    * Those two hypotheses are separated by reversing the chain while KEEPING the
    * dependency: queue the draw first, so it sits earlier in the next-pointer
    * chain, and point its dependency at the helper that has not been added yet.
    *
    * pan_jc_add_job assigns index = ++jc->job_index, so with the draw added
    * first the helper is deterministically two past the current value. This is a
    * forward dependency, which is the point: if the hardware honours it the
    * output stays correct and the dependency is doing real work, and if it does
    * not the draw reads the CPU placeholder and the output collapses to the
    * degenerate case.
    */
   const bool chain_reverse = is_indirect && pan_env_flag("PANVK_INDIRECT_CHAIN_REVERSE");

   if (chain_reverse) {
      const unsigned predicted_helper_index = batch->vtc_jc.job_index + 2;

      pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_NOT_STARTED,
                     batch->ubo_gather.hdr != NULL, false, 0,
                     predicted_helper_index, &draw->jobs.idvs, false);

      struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
      if (draw->info.index.index_size) {
         panlib_draw_indexed_indirect_helper(
            &precomp_ctx, panlib_1d(1), PANLIB_BARRIER_JM_SUPPRESS_PREFETCH,
            draw->info.indirect.buffer_dev_addr, draw->jobs.idvs.gpu,
            draw->info.index.buffer_dev_addr, draw->info.index.index_size,
            fv_sysval, bi_sysval, vs_attribs, inst_mask, inst_strides,
            draw->info.indirect.count_buffer_dev_addr, draw->info.draw_id);
      } else {
         panlib_draw_indirect_helper(&precomp_ctx, panlib_1d(1),
                                     PANLIB_BARRIER_JM_SUPPRESS_PREFETCH,
                                     draw->info.indirect.buffer_dev_addr,
                                     draw->jobs.idvs.gpu, fv_sysval, bi_sysval,
                                     vs_attribs, inst_mask, inst_strides,
                                     draw->info.indirect.count_buffer_dev_addr,
                                     draw->info.draw_id);
      }

      /* Confirm the prediction rather than trusting it. If the helper did not
       * land on the predicted index the experiment is measuring the wrong thing
       * and the log has to say so. */
      if (batch->vtc_jc.job_index != predicted_helper_index) {
         PAN_V9_TRACE(
                 "[PANVK_DEBUG_CHAINREV] prediction MISSED: expected helper "
                 "index %u, got %u\n",
                 predicted_helper_index, batch->vtc_jc.job_index);
      } else {
         PAN_V9_TRACE(
                 "[PANVK_DEBUG_CHAINREV] draw queued first, forward dep on "
                 "helper index %u (confirmed)\n",
                 predicted_helper_index);
      }
      goto indirect_chain_done;
   }

   if (is_indirect) {
      struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);

      if (draw->info.index.index_size) {
         panlib_draw_indexed_indirect_helper(
            &precomp_ctx, panlib_1d(1), PANLIB_BARRIER_JM_SUPPRESS_PREFETCH,
            draw->info.indirect.buffer_dev_addr, draw->jobs.idvs.gpu,
            draw->info.index.buffer_dev_addr, draw->info.index.index_size,
            fv_sysval, bi_sysval, vs_attribs, inst_mask, inst_strides,
            draw->info.indirect.count_buffer_dev_addr, draw->info.draw_id);
      } else {
         panlib_draw_indirect_helper(&precomp_ctx, panlib_1d(1),
                                     PANLIB_BARRIER_JM_SUPPRESS_PREFETCH,
                                     draw->info.indirect.buffer_dev_addr,
                                     draw->jobs.idvs.gpu, fv_sysval, bi_sysval,
                                     vs_attribs, inst_mask, inst_strides,
                                     draw->info.indirect.count_buffer_dev_addr,
                                     draw->info.draw_id);
      }

      indirect_dep = batch->vtc_jc.job_index;

      /* Negative control, T4.4.9. Dropping the dependency leaves the helper and
       * the draw job unordered. If the correct output still appears, then the
       * ordering was never coming from this dependency and every indirect
       * result would be a race that happens to resolve favourably. Env-gated so
       * the shipped path is untouched. */
      if (pan_env_flag("PANVK_INDIRECT_NODEP"))
         indirect_dep = 0;
   }

   /* PATCH v9: a parked indirect draw is a tiler job too, but pan_jc only
    * tracks jobs whose type uses tiling, and NOT_STARTED does not. So
    * first_tiler stayed NULL in a batch holding only indirect draws, and
    * cmd_close_batch() emitted no fragment job for it unless the render pass
    * was ending with a store. A batch closed inside a render pass (end of a
    * query, a job-index split) then dropped all its indirect draws: every
    * tessellated or GS draw inside a pipeline statistics query rendered
    * nothing (CTS query_pool.statistics_query.tes_*, clipping_* with
    * tessellation). The job was also outside the tiler dependency chain, so
    * it was not ordered against the draws around it. Track it like
    * pan_jc_add_job() tracks MALLOC_VERTEX: wait on the previous tiler job
    * (dependency 1, dependency 2 is the helper) and become the new tail.
    * PANVK_V9_INDIRECT_UNTRACKED=1 restores the old behaviour (control). */
   static int indirect_untracked = -1;
   if (indirect_untracked < 0)
      indirect_untracked = pan_env_flag("PANVK_V9_INDIRECT_UNTRACKED") != NULL;
   const bool track_parked = is_indirect && !indirect_untracked;
   const unsigned prev_tiler_dep = batch->vtc_jc.tiler_dep;

   /* PANVK_UBO_PUSH: the draw reads FAU words that the batch's gather job
    * writes. Dependency 1 is free on direct draws; otherwise use a barrier. */
   const unsigned ubo_dep = batch->ubo_gather.hdr ? batch->ubo_gather.job_index
                                                  : 0;
   const unsigned local_dep = track_parked ? prev_tiler_dep : ubo_dep;
   const bool ubo_barrier = ubo_dep && local_dep != ubo_dep;

   pan_jc_add_job(&batch->vtc_jc,
                  is_indirect ? MALI_JOB_TYPE_NOT_STARTED
                              : MALI_JOB_TYPE_MALLOC_VERTEX,
                  ubo_barrier, false, local_dep,
                  indirect_dep, &draw->jobs.idvs, false);

   if (track_parked) {
      if (!batch->vtc_jc.first_tiler) {
         batch->vtc_jc.first_tiler = (void *)draw->jobs.idvs.cpu;
         batch->vtc_jc.first_tiler_dep1 = prev_tiler_dep;
      }
      batch->vtc_jc.tiler_dep = batch->vtc_jc.job_index;
   }

indirect_chain_done:

   /* PATCH DEBUG v9 (sementara): sesudah pan_jc_add_job, jadi Job Header
    * sudah terisi. Lihat komentar di panvk_debug_dump_malloc_vertex_job(). */
   panvk_debug_dump_malloc_vertex_job(&draw->jobs.idvs, "sesudah pan_jc_add_job");

   PAN_V9_TRACE(
           "[PANVK_DEBUG_MVJ] jc state: first_job=0x%" PRIx64
           " job_index=%u tiler_dep=%u\n",
           batch->vtc_jc.first_job, batch->vtc_jc.job_index,
           batch->vtc_jc.tiler_dep);

   clear_dirty_after_draw(cmdbuf);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDraw)(VkCommandBuffer commandBuffer, uint32_t vertexCount,
                        uint32_t instanceCount, uint32_t firstVertex,
                        uint32_t firstInstance)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (instanceCount == 0 || vertexCount == 0)
      return;

   /* gl_BaseVertexARB is a signed integer, and it should expose the value of
    * firstVertex in a non-indexed draw. */
   assert(firstVertex < INT32_MAX);

   /* gl_BaseInstance is a signed integer, and it should expose the value of
    * firstInstance. */
   assert(firstInstance < INT32_MAX);

   struct panvk_draw_data draw = {
      .info = {
         .vertex.base = firstVertex,
         .vertex.count = vertexCount,
         .instance.base = firstInstance,
         .instance.count = instanceCount,
         .prim = panvk_get_client_prim(cmdbuf),
      },
      .vertex_range = vertexCount,
   };

   panvk_cmd_draw_v9(cmdbuf, &draw);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndexed)(VkCommandBuffer commandBuffer,
                               uint32_t indexCount, uint32_t instanceCount,
                               uint32_t firstIndex, int32_t vertexOffset,
                               uint32_t firstInstance)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (instanceCount == 0 || indexCount == 0)
      return;

   assert(firstInstance < INT32_MAX);

   struct panvk_draw_data draw = {
      .info = {
         .index = panvk_draw_info_index(cmdbuf, firstIndex),
         .vertex.base = vertexOffset,
         .vertex.count = indexCount,
         .instance.base = firstInstance,
         .instance.count = instanceCount,
         .prim = panvk_get_client_prim(cmdbuf),
      },
      .vertex_range = indexCount,
   };

   panvk_cmd_draw_v9(cmdbuf, &draw);
}

/* Job Manager hardware has no indirect-draw descriptor, so each draw becomes a
 * job plus a helper kernel that patches its counts. There is no multi-draw
 * descriptor either, so drawCount (or maxDrawCount) is unrolled into that many
 * pairs, each with its own gl_DrawID. For the *Count variants the helper turns
 * the draws at or past the GPU-side count into NULL jobs.
 *
 * The counts in the draw info are placeholders. The helper overwrites
 * index_count, base_vertex_offset and the instance count in the descriptor
 * before the job is allowed to run.
 *
 * The indexed variant does NOT run the Bifrost min/max index search: on v9
 * varyings are hardware-allocated, so nothing is sized from the vertex range. */
static void
draw_indirect_unrolled_v9(struct panvk_cmd_buffer *cmdbuf,
                          struct panvk_buffer *buffer, VkDeviceSize offset,
                          uint32_t draw_count, uint32_t stride, bool indexed,
                          uint64_t count_buffer_dev_addr)
{
   /* Negative controls: PANVK_MDI_NO_DRAWID gives every draw gl_DrawID 0,
    * PANVK_MDI_NO_COUNT ignores the count buffer (all maxDrawCount draws
    * run). */
   const bool no_draw_id = pan_env_flag("PANVK_MDI_NO_DRAWID") != NULL;
   if (pan_env_flag("PANVK_MDI_NO_COUNT"))
      count_buffer_dev_addr = 0;

   for (uint32_t i = 0; i < draw_count; i++) {
      struct panvk_draw_data draw = {
         .info = {
            .vertex.base = 0,
            .vertex.count = 1,
            .instance.base = 0,
            .instance.count = 1,
            .prim = panvk_get_client_prim(cmdbuf),
            .indirect.buffer_dev_addr =
               panvk_buffer_gpu_ptr(buffer, offset + (uint64_t)i * stride),
            .indirect.count_buffer_dev_addr = count_buffer_dev_addr,
            .indirect.draw_count = 1,
            .indirect.stride = 0,
            .draw_id = no_draw_id ? 0 : i,
         },
         .vertex_range = 1,
      };
      if (indexed)
         draw.info.index =
            (__typeof__(draw.info.index))panvk_draw_info_index(cmdbuf, 0);

      panvk_cmd_draw_v9(cmdbuf, &draw);
   }
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndirect)(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                                VkDeviceSize offset, uint32_t drawCount,
                                uint32_t stride)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   VK_FROM_HANDLE(panvk_buffer, buffer, _buffer);

   if (drawCount == 0)
      return;

   draw_indirect_unrolled_v9(cmdbuf, buffer, offset, drawCount, stride, false,
                             0);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndexedIndirect)(VkCommandBuffer commandBuffer,
                                       VkBuffer _buffer, VkDeviceSize offset,
                                       uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   VK_FROM_HANDLE(panvk_buffer, buffer, _buffer);

   if (drawCount == 0 || cmdbuf->state.gfx.ib.size == 0)
      return;

   draw_indirect_unrolled_v9(cmdbuf, buffer, offset, drawCount, stride, true,
                             0);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndirectCount)(VkCommandBuffer commandBuffer,
                                     VkBuffer _buffer, VkDeviceSize offset,
                                     VkBuffer countBuffer,
                                     VkDeviceSize countBufferOffset,
                                     uint32_t maxDrawCount, uint32_t stride)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   VK_FROM_HANDLE(panvk_buffer, buffer, _buffer);
   VK_FROM_HANDLE(panvk_buffer, count_buffer, countBuffer);

   if (maxDrawCount == 0)
      return;

   draw_indirect_unrolled_v9(cmdbuf, buffer, offset, maxDrawCount, stride,
                             false,
                             panvk_buffer_gpu_ptr(count_buffer,
                                                  countBufferOffset));
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndexedIndirectCount)(VkCommandBuffer commandBuffer,
                                            VkBuffer _buffer,
                                            VkDeviceSize offset,
                                            VkBuffer countBuffer,
                                            VkDeviceSize countBufferOffset,
                                            uint32_t maxDrawCount,
                                            uint32_t stride)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   VK_FROM_HANDLE(panvk_buffer, buffer, _buffer);
   VK_FROM_HANDLE(panvk_buffer, count_buffer, countBuffer);

   if (maxDrawCount == 0 || cmdbuf->state.gfx.ib.size == 0)
      return;

   draw_indirect_unrolled_v9(cmdbuf, buffer, offset, maxDrawCount, stride,
                             true,
                             panvk_buffer_gpu_ptr(count_buffer,
                                                  countBufferOffset));
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBeginRendering)(VkCommandBuffer commandBuffer,
                                  const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   struct panvk_cmd_graphics_state *state = &cmdbuf->state.gfx;
   bool resuming = pRenderingInfo->flags & VK_RENDERING_RESUMING_BIT;

   if (resuming && cmdbuf->cur_batch) {
      state->render.flags = pRenderingInfo->flags;
   } else {
      if (cmdbuf->cur_batch)
         panvk_per_arch(cmd_close_batch)(cmdbuf);

      panvk_per_arch(cmd_init_render_state)(cmdbuf, pRenderingInfo);
      cmdbuf->state.gfx.render.fb.needs_load = !resuming;
   }

   if (!cmdbuf->cur_batch)
      panvk_per_arch(cmd_open_batch)(cmdbuf);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdEndRendering)(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (!(cmdbuf->state.gfx.render.flags & VK_RENDERING_SUSPENDING_BIT)) {
      const struct pan_fb_load *fb_load = &cmdbuf->state.gfx.render.fb.load;
      bool always_load = fb_load->z.always || fb_load->s.always;
      for (unsigned rt = 0; rt < PAN_MAX_RTS; rt++) {
         if (fb_load->rts[rt].always)
            always_load = true;
      }

      if (always_load)
         panvk_per_arch(cmd_alloc_fb_desc)(cmdbuf);

      cmdbuf->state.gfx.render.fb.needs_store = true;

      panvk_per_arch(cmd_close_batch)(cmdbuf);
      cmdbuf->cur_batch = NULL;
      panvk_per_arch(cmd_meta_resolve_attachments)(cmdbuf);
      cmdbuf->state.gfx.render.view_mask = 0;
   }
}

#endif /* PAN_ARCH >= 9 */
