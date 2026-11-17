/*
 * Copyright © 2024 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_QUERY_POOL_H
#define PANVK_QUERY_POOL_H

#ifndef PAN_ARCH
#error "PAN_ARCH must be defined"
#endif

#include <stdint.h>

#include "panvk_mempool.h"
#include "vk_query_pool.h"

#if PAN_ARCH >= 10
#include "panvk_cmd_buffer.h"
#endif

#if PAN_ARCH == 9
#include "panvk_instance.h"

/* PATCH v9: timestamp query pools and writes are always implemented.
 * Creating a TIMESTAMP pool is valid whatever timestampValidBits says, and
 * layers (DXVK, wrappers) write timestamps even when the queue reports 0
 * valid bits. Without an implementation the pool creation reached
 * UNREACHABLE (undefined behaviour in release builds: hang or crash) and
 * written queries never became available.
 *
 * What PANVK_V9_TIMESTAMP=1 / pan_v9_timestamps select is only the
 * advertisement (timestampValidBits, timestampComputeAndGraphics), see
 * panvk_v9_timestamp_queries_enabled(). */
static inline bool
panvk_v9_timestamps(const struct vk_device *dev)
{
   (void)dev;
   return true;
}

/* Timestamps are written but not advertised: log once, for diagnosis. */
static inline bool
panvk_v9_timestamps_advertised(const struct vk_device *dev)
{
   const struct panvk_instance *instance =
      to_panvk_instance(dev->physical->instance);

   return pan_env_flag("PANVK_V9_TIMESTAMP") != NULL ||
          instance->drirc.misc.v9_timestamps;
}
#endif

struct panvk_query_report {
   uint64_t value;
};

struct panvk_query_available_obj {
#if PAN_ARCH >= 10
   struct panvk_cs_sync32 sync_obj;
#else
   uint32_t value;
#endif
};

static_assert(sizeof(struct panvk_query_report) == 8,
              "panvk_query_report size is expected to be 8");

struct panvk_query_pool {
   struct vk_query_pool vk;

   uint32_t query_stride;
   uint32_t reports_per_query;

   struct panvk_priv_mem mem;
   struct panvk_priv_mem available_mem;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(panvk_query_pool, vk.base, VkQueryPool,
                               VK_OBJECT_TYPE_QUERY_POOL)

static inline uint32_t
panvk_query_available_offset(struct panvk_query_pool *pool, uint32_t query)
{
   assert(query < pool->vk.query_count);
   return query * sizeof(struct panvk_query_available_obj);
}

static inline uint64_t
panvk_query_available_dev_addr(struct panvk_query_pool *pool, uint32_t query)
{
   return panvk_priv_mem_dev_addr(pool->available_mem) +
          panvk_query_available_offset(pool, query);
}

static inline uint64_t
panvk_query_offset(struct panvk_query_pool *pool, uint32_t query)
{
   assert(query < pool->vk.query_count);
   return query * (uint64_t)pool->query_stride;
}

static inline uint64_t
panvk_query_report_dev_addr(struct panvk_query_pool *pool, uint32_t query)
{
   return panvk_priv_mem_dev_addr(pool->mem) + panvk_query_offset(pool, query);
}

#endif
