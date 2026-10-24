/* Moreau stand-in for SuiteSparse_config.h (SuiteSparse 7.14.1).
 *
 * The other files in this directory are AMD 3.3.4 from SuiteSparse 7.14.1,
 * unmodified: the headers, and the sources amd_order calls. They need only
 * these definitions from SuiteSparse_config, so this header replaces it and
 * its library. Debug printing (SUITESPARSE_PRINTF) is left out because AMD
 * is always compiled with NDEBUG. */

#ifndef SUITESPARSE_CONFIG_H
#define SUITESPARSE_CONFIG_H

#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define SUITESPARSE_VER_CODE(main, sub) ((main) * 1000 + (sub))
#define SUITESPARSE__VERCODE(main, sub, patch) (((main) * 1000ULL + (sub)) * 1000ULL + (patch))
#define SUITESPARSE__VERSION SUITESPARSE__VERCODE(7, 14, 1)

/* As in SuiteSparse_config.c: allocate at least one byte. amd_order checks
 * n * size for overflow before calling. */
static inline void *SuiteSparse_malloc(size_t n, size_t size) {
    return malloc((n > 0 ? n : 1) * (size > 0 ? size : 1));
}

static inline void *SuiteSparse_free(void *p) {
    free(p);
    return NULL;
}

#endif
