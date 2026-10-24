/* hipmem-shim-01: LD_PRELOAD in front of PyTorch's HIP virtual-memory calls.
   SHIM_MODE=log       pass everything through, count calls
   SHIM_MODE=recycle   hipMemRelease keeps the physical handle in a per-size pool; hipMemCreate of that size reuses it
   SHIM_MODE=norelease hipMemRelease is dropped (memory leaks; diagnostic only)
   Counters go to stderr at exit. Single-device use only. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int hipError_t; typedef void *hipHandle;
static hipError_t (*r_create)(hipHandle *, size_t, const void *, unsigned long long);
static hipError_t (*r_release)(hipHandle);
static hipError_t (*r_map)(void *, size_t, size_t, hipHandle, unsigned long long);
static hipError_t (*r_unmap)(void *, size_t);
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int mode = -1; /* 0 log, 1 recycle, 2 norelease */
#define MAXH 65536
static hipHandle hh[MAXH]; static size_t hs[MAXH]; static int nh;   /* live handle -> size */
static hipHandle ph[MAXH]; static size_t ps[MAXH]; static int np;   /* recycle pool */
static long c_create, c_create_real, c_create_fail, c_create_pool, c_release, c_release_real, c_map, c_unmap;
static void init(void) {
  if (mode >= 0) return;
  r_create = dlvsym(RTLD_NEXT, "hipMemCreate", "hip_5.1"); r_release = dlvsym(RTLD_NEXT, "hipMemRelease", "hip_5.1");
  r_map = dlvsym(RTLD_NEXT, "hipMemMap", "hip_5.1"); r_unmap = dlvsym(RTLD_NEXT, "hipMemUnmap", "hip_5.1");
  if (!r_create) r_create = dlsym(RTLD_NEXT, "hipMemCreate"); if (!r_release) r_release = dlsym(RTLD_NEXT, "hipMemRelease");
  if (!r_map) r_map = dlsym(RTLD_NEXT, "hipMemMap"); if (!r_unmap) r_unmap = dlsym(RTLD_NEXT, "hipMemUnmap");
  const char *m = getenv("SHIM_MODE"); mode = !m ? 0 : !strcmp(m, "recycle") ? 1 : !strcmp(m, "norelease") ? 2 : 0;
  fprintf(stderr, "SHIM active, mode %d, real fns %s\n", mode, (r_create && r_release && r_map && r_unmap) ? "found" : "MISSING");
}
__attribute__((destructor)) static void fini(void) {
  fprintf(stderr, "SHIM counts: create %ld (real %ld, failed %ld, from pool %ld), release %ld (real %ld), map %ld, unmap %ld, pool left %d\n",
          c_create, c_create_real, c_create_fail, c_create_pool, c_release, c_release_real, c_map, c_unmap, np);
}
hipError_t hipMemCreate(hipHandle *h, size_t size, const void *prop, unsigned long long flags) {
  pthread_mutex_lock(&mu); init(); c_create++;
  if (mode == 1) for (int i = np - 1; i >= 0; i--) if (ps[i] == size) {
    *h = ph[i]; ph[i] = ph[np - 1]; ps[i] = ps[np - 1]; np--; c_create_pool++;
    if (nh < MAXH) { hh[nh] = *h; hs[nh] = size; nh++; }
    pthread_mutex_unlock(&mu); return 0; }
  pthread_mutex_unlock(&mu);
  hipError_t e = r_create(h, size, prop, flags);
  pthread_mutex_lock(&mu);
  if (e == 0) { c_create_real++; if (nh < MAXH) { hh[nh] = *h; hs[nh] = size; nh++; } } else c_create_fail++;
  pthread_mutex_unlock(&mu); return e;
}
hipError_t hipMemRelease(hipHandle h) {
  pthread_mutex_lock(&mu); init(); c_release++;
  size_t size = 0; int found = 0;
  for (int i = 0; i < nh; i++) if (hh[i] == h) { size = hs[i]; hh[i] = hh[nh - 1]; hs[i] = hs[nh - 1]; nh--; found = 1; break; }
  if (mode == 1 && found && np < MAXH) { ph[np] = h; ps[np] = size; np++; pthread_mutex_unlock(&mu); return 0; }
  if (mode == 2) { pthread_mutex_unlock(&mu); return 0; }
  c_release_real++; pthread_mutex_unlock(&mu); return r_release(h);
}
hipError_t hipMemMap(void *p, size_t s, size_t o, hipHandle h, unsigned long long f) {
  pthread_mutex_lock(&mu); init(); c_map++; pthread_mutex_unlock(&mu); return r_map(p, s, o, h, f); }
hipError_t hipMemUnmap(void *p, size_t s) {
  pthread_mutex_lock(&mu); init(); c_unmap++; pthread_mutex_unlock(&mu); return r_unmap(p, s); }
