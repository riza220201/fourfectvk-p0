/*
 * Copyright 2026 Noysz
 * SPDX-License-Identifier: MIT
 */

#ifndef PAN_ENV_H
#define PAN_ENV_H

#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* Debug and control knobs that switch something on by being set.
 * Unset, empty, "0", "false", "off" and "no" all mean off, so KNOB=0 turns
 * a knob off instead of on. Returns the value, or NULL when off. */
static inline const char *
pan_env_flag(const char *name)
{
   const char *v = getenv(name);

   if (!v || !v[0] || !strcmp(v, "0") || !strcasecmp(v, "false") ||
       !strcasecmp(v, "off") || !strcasecmp(v, "no"))
      return NULL;

   return v;
}

#endif /* PAN_ENV_H */
