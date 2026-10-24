/* hipmem-shim-02: LD_PRELOAD in front of PyTorch's HIP virtual-memory calls (libc10_hip.so imports them @hip_5.1).
   SHIM_MODE=recycle   a physical handle PyTorch releases is kept in a pool keyed by size and allocation properties
                       (type, handle type, location type, card id); PyTorch's next hipMemCreate with the same key gets it back.
                       The runtime never sees a release from PyTorch, so it never frees or hands that memory out again.
   SHIM_MODE=log       pass everything through, count calls (the control).
   Only calls made from libc10_hip.so are pooled; every other caller (RCCL, libtorch_hip, ...) passes straight through.
   Counts go to stderr at start, every 256 pooled events and at exit, tagged with the pid. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
typedef int hipError_t; typedef void *hipHandle;
static hipError_t (*r_create)(hipHandle *, size_t, const void *, unsigned long long);
static hipError_t (*r_release)(hipHandle);
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int mode = -1; /* 0 log, 1 recycle */
#define MAXH 65536
typedef struct { hipHandle h; size_t size; unsigned char key[16]; } ent;
static ent live[MAXH]; static int nlive;   /* handles created by libc10_hip */
static ent pool[MAXH]; static int npool;
static long c_create, c_create_real, c_create_fail, c_create_pool, c_create_other, c_release, c_release_real, c_release_other, c_dev[16];
static void stats(const char *why) {
  fprintf(stderr, "[hipmem-shim-02] pid %d %s: mode %s, torch creates %ld (real %ld, failed %ld, from pool %ld), other creates %ld, "
          "torch releases %ld (to runtime %ld), other releases %ld, live %d, pool %d, pool reuse by card:", getpid(), why,
          mode == 1 ? "recycle" : "log", c_create, c_create_real, c_create_fail, c_create_pool, c_create_other, c_release, c_release_real,
          c_release_other, nlive, npool);
  for (int i = 0; i < 16; i++) if (c_dev[i]) fprintf(stderr, " %d=%ld", i, c_dev[i]);
  fprintf(stderr, "\n");
}
static void init(void) {
  if (mode >= 0) return;
  r_create = dlvsym(RTLD_NEXT, "hipMemCreate", "hip_5.1"); r_release = dlvsym(RTLD_NEXT, "hipMemRelease", "hip_5.1");
  if (!r_create) r_create = dlsym(RTLD_NEXT, "hipMemCreate"); if (!r_release) r_release = dlsym(RTLD_NEXT, "hipMemRelease");
  const char *m = getenv("SHIM_MODE"); mode = (m && !strcmp(m, "recycle")) ? 1 : 0;
  stats(r_create && r_release ? "start (real functions found)" : "start (REAL FUNCTIONS MISSING)");
}
__attribute__((destructor)) static void fini(void) { if (mode >= 0) stats("exit"); }
static int from_torch(void *ra) { Dl_info i; return dladdr(ra, &i) && i.dli_fname && strstr(i.dli_fname, "libc10_hip.so"); }
hipError_t hipMemCreate(hipHandle *h, size_t size, const void *prop, unsigned long long flags) {
  void *ra = __builtin_return_address(0); int t = from_torch(ra);
  pthread_mutex_lock(&mu); init();
  if (!t) { c_create_other++; pthread_mutex_unlock(&mu); return r_create(h, size, prop, flags); }
  c_create++; unsigned char key[16]; memset(key, 0, 16); if (prop) memcpy(key, prop, 16);
  if (mode == 1) for (int i = npool - 1; i >= 0; i--) if (pool[i].size == size && !memcmp(pool[i].key, key, 16)) {
    *h = pool[i].h; if (nlive < MAXH) live[nlive++] = pool[i]; pool[i] = pool[--npool]; c_create_pool++;
    int dev; memcpy(&dev, key + 12, 4); if (dev >= 0 && dev < 16) c_dev[dev]++;
    if (c_create_pool % 256 == 0) stats("progress");
    pthread_mutex_unlock(&mu); return 0; }
  pthread_mutex_unlock(&mu);
  hipError_t e = r_create(h, size, prop, flags);
  pthread_mutex_lock(&mu);
  if (e == 0) { c_create_real++; if (nlive < MAXH) { live[nlive].h = *h; live[nlive].size = size; memcpy(live[nlive].key, key, 16); nlive++; } }
  else c_create_fail++;
  pthread_mutex_unlock(&mu); return e;
}
hipError_t hipMemRelease(hipHandle h) {
  pthread_mutex_lock(&mu); init();
  int f = -1; for (int i = 0; i < nlive; i++) if (live[i].h == h) { f = i; break; }
  if (f < 0) { c_release_other++; pthread_mutex_unlock(&mu); return r_release(h); }
  c_release++;
  if (mode == 1 && npool < MAXH) { pool[npool++] = live[f]; live[f] = live[--nlive]; if (c_release % 256 == 0) stats("progress"); pthread_mutex_unlock(&mu); return 0; }
  live[f] = live[--nlive]; c_release_real++; pthread_mutex_unlock(&mu); return r_release(h);
}
