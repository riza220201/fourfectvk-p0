#include "panvk_kbase_owner.h"

#include <assert.h>
#include <stdlib.h>

#include "c11/threads.h"
#include "util/cnd_monotonic.h"
#include "util/u_atomic.h"

struct panvk_kbase_owner {
   mtx_t mutex;
   struct u_cnd_monotonic cond;
   int refs;
   unsigned active;
   void *data;
};

struct panvk_kbase_owner *
panvk_kbase_owner_create(void *data)
{
   struct panvk_kbase_owner *owner = calloc(1, sizeof(*owner));
   if (!owner)
      return NULL;
   if (mtx_init(&owner->mutex, mtx_plain) != thrd_success)
      goto fail;
   if (u_cnd_monotonic_init(&owner->cond) != thrd_success) {
      mtx_destroy(&owner->mutex);
      goto fail;
   }
   owner->refs = 1;
   owner->data = data;
   return owner;
fail:
   free(owner);
   return NULL;
}

struct panvk_kbase_owner *
panvk_kbase_owner_ref(struct panvk_kbase_owner *owner)
{
   if (owner)
      p_atomic_inc(&owner->refs);
   return owner;
}

void
panvk_kbase_owner_unref(struct panvk_kbase_owner *owner)
{
   if (!owner || !p_atomic_dec_zero(&owner->refs))
      return;
   assert(!owner->active);
   u_cnd_monotonic_destroy(&owner->cond);
   mtx_destroy(&owner->mutex);
   free(owner);
}

void *
panvk_kbase_owner_enter(struct panvk_kbase_owner *owner)
{
   if (!owner)
      return NULL;
   mtx_lock(&owner->mutex);
   void *data = owner->data;
   if (data) {
      owner->active++;
      panvk_kbase_owner_ref(owner);
   }
   mtx_unlock(&owner->mutex);
   return data;
}

void
panvk_kbase_owner_leave(struct panvk_kbase_owner *owner)
{
   mtx_lock(&owner->mutex);
   assert(owner->active);
   if (!--owner->active)
      u_cnd_monotonic_broadcast(&owner->cond);
   mtx_unlock(&owner->mutex);
   panvk_kbase_owner_unref(owner);
}

void
panvk_kbase_owner_close(struct panvk_kbase_owner *owner)
{
   mtx_lock(&owner->mutex);
   owner->data = NULL;
   while (owner->active)
      u_cnd_monotonic_wait(&owner->cond, &owner->mutex);
   mtx_unlock(&owner->mutex);
}
