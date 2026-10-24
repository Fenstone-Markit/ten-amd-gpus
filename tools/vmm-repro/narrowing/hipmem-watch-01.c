/* hipmem-watch-01: diagnostic LD_PRELOAD, passes every call through. It watches the first handle mapped in the process (where
   vmm-repro-03 puts its four marked blocks: the first 2 MiB small-pool page) and prints any hipMemUnmap that overlaps that
   page's addresses and any hipMemRelease of that handle, plus a count of every call. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
typedef int hipError_t; typedef void *hipHandle;
static hipError_t (*r_create)(hipHandle *, size_t, const void *, unsigned long long);
static hipError_t (*r_release)(hipHandle);
static hipError_t (*r_map)(void *, size_t, size_t, hipHandle, unsigned long long);
static hipError_t (*r_unmap)(void *, size_t);
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int ready; static uintptr_t w0, w1; static hipHandle wh; static size_t wsize;
static long c_create, c_release, c_map, c_unmap, w_unmap, w_release, w_remap;
static void init(void) {
  if (ready) return; ready = 1;
  r_create = dlvsym(RTLD_NEXT, "hipMemCreate", "hip_5.1"); r_release = dlvsym(RTLD_NEXT, "hipMemRelease", "hip_5.1");
  r_map = dlvsym(RTLD_NEXT, "hipMemMap", "hip_5.1"); r_unmap = dlvsym(RTLD_NEXT, "hipMemUnmap", "hip_5.1");
  fprintf(stderr, "[hipmem-watch-01] active, real functions %s\n", (r_create && r_release && r_map && r_unmap) ? "found" : "MISSING");
}
__attribute__((destructor)) static void fini(void) {
  fprintf(stderr, "[hipmem-watch-01] counts: create %ld, release %ld, map %ld, unmap %ld; watched page 0x%lx-0x%lx (handle %p, %zu bytes): "
          "unmaps overlapping it %ld, releases of its handle %ld, other handles mapped onto it %ld\n",
          c_create, c_release, c_map, c_unmap, (unsigned long)w0, (unsigned long)w1, wh, wsize, w_unmap, w_release, w_remap);
}
hipError_t hipMemCreate(hipHandle *h, size_t s, const void *p, unsigned long long f) {
  pthread_mutex_lock(&mu); init(); c_create++; pthread_mutex_unlock(&mu); return r_create(h, s, p, f); }
hipError_t hipMemMap(void *ptr, size_t s, size_t o, hipHandle h, unsigned long long f) {
  pthread_mutex_lock(&mu); init(); c_map++; uintptr_t a = (uintptr_t)ptr;
  if (!w0) { w0 = a; w1 = a + s; wh = h; wsize = s; fprintf(stderr, "[hipmem-watch-01] watching first map 0x%lx +%zu handle %p\n", (unsigned long)a, s, h); }
  else if (a < w1 && a + s > w0) { w_remap++; fprintf(stderr, "[hipmem-watch-01] MAP onto watched page: 0x%lx +%zu handle %p\n", (unsigned long)a, s, h); }
  pthread_mutex_unlock(&mu); return r_map(ptr, s, o, h, f); }
hipError_t hipMemUnmap(void *ptr, size_t s) {
  pthread_mutex_lock(&mu); init(); c_unmap++; uintptr_t a = (uintptr_t)ptr;
  if (w0 && a < w1 && a + s > w0) { w_unmap++; fprintf(stderr, "[hipmem-watch-01] UNMAP overlapping watched page: 0x%lx +%zu\n", (unsigned long)a, s); }
  pthread_mutex_unlock(&mu); return r_unmap(ptr, s); }
hipError_t hipMemRelease(hipHandle h) {
  pthread_mutex_lock(&mu); init(); c_release++;
  if (wh && h == wh) { w_release++; fprintf(stderr, "[hipmem-watch-01] RELEASE of watched handle %p\n", h); }
  pthread_mutex_unlock(&mu); return r_release(h); }
