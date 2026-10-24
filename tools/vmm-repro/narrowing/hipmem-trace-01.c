/* hipmem-trace-01: diagnostic LD_PRELOAD; passes every call through and writes one line per HIP virtual-memory call to $TRACE_FILE:
   V addr size (reserve)  F addr size (address free)  C handle size type htype ltype id (create)  M addr size offset handle (map)
   S addr size (set access)  U addr size (unmap)  R handle (release). Return codes other than 0 are appended as "e=N". */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int hipError_t; typedef void *hipHandle;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static FILE *tf;
#define REAL(name, ...) static hipError_t (*r_##name)(__VA_ARGS__); 
REAL(hipMemAddressReserve, void **, size_t, size_t, void *, unsigned long long)
REAL(hipMemAddressFree, void *, size_t)
REAL(hipMemCreate, hipHandle *, size_t, const void *, unsigned long long)
REAL(hipMemMap, void *, size_t, size_t, hipHandle, unsigned long long)
REAL(hipMemSetAccess, void *, size_t, const void *, size_t)
REAL(hipMemUnmap, void *, size_t)
REAL(hipMemRelease, hipHandle)
static void *get(const char *n) { void *f = dlvsym(RTLD_NEXT, n, "hip_5.1"); return f ? f : dlsym(RTLD_NEXT, n); }
static void init(void) {
  if (tf) return;
  r_hipMemAddressReserve = get("hipMemAddressReserve"); r_hipMemAddressFree = get("hipMemAddressFree"); r_hipMemCreate = get("hipMemCreate");
  r_hipMemMap = get("hipMemMap"); r_hipMemSetAccess = get("hipMemSetAccess"); r_hipMemUnmap = get("hipMemUnmap"); r_hipMemRelease = get("hipMemRelease");
  const char *p = getenv("TRACE_FILE"); tf = fopen(p ? p : "/tmp/hipmem-trace.txt", "w"); if (!tf) tf = stderr;
}
#define LOG(...) do { fprintf(tf, __VA_ARGS__); if (e) fprintf(tf, " e=%d", e); fprintf(tf, "\n"); fflush(tf); } while (0)
hipError_t hipMemAddressReserve(void **p, size_t s, size_t al, void *addr, unsigned long long f) {
  pthread_mutex_lock(&mu); init(); hipError_t e = r_hipMemAddressReserve(p, s, al, addr, f); LOG("V %p %zu", e ? NULL : *p, s); pthread_mutex_unlock(&mu); return e; }
hipError_t hipMemAddressFree(void *p, size_t s) {
  pthread_mutex_lock(&mu); init(); hipError_t e = r_hipMemAddressFree(p, s); LOG("F %p %zu", p, s); pthread_mutex_unlock(&mu); return e; }
hipError_t hipMemCreate(hipHandle *h, size_t s, const void *prop, unsigned long long f) {
  pthread_mutex_lock(&mu); init(); hipError_t e = r_hipMemCreate(h, s, prop, f); int k[4] = {0, 0, 0, 0}; if (prop) memcpy(k, prop, 16);
  LOG("C %p %zu %d %d %d %d", e ? NULL : *h, s, k[0], k[1], k[2], k[3]); pthread_mutex_unlock(&mu); return e; }
hipError_t hipMemMap(void *p, size_t s, size_t o, hipHandle h, unsigned long long f) {
  pthread_mutex_lock(&mu); init(); hipError_t e = r_hipMemMap(p, s, o, h, f); LOG("M %p %zu %zu %p", p, s, o, h); pthread_mutex_unlock(&mu); return e; }
hipError_t hipMemSetAccess(void *p, size_t s, const void *d, size_t n) {
  pthread_mutex_lock(&mu); init(); hipError_t e = r_hipMemSetAccess(p, s, d, n); LOG("S %p %zu", p, s); pthread_mutex_unlock(&mu); return e; }
hipError_t hipMemUnmap(void *p, size_t s) {
  pthread_mutex_lock(&mu); init(); hipError_t e = r_hipMemUnmap(p, s); LOG("U %p %zu", p, s); pthread_mutex_unlock(&mu); return e; }
hipError_t hipMemRelease(hipHandle h) {
  pthread_mutex_lock(&mu); init(); hipError_t e = r_hipMemRelease(h); LOG("R %p", h); pthread_mutex_unlock(&mu); return e; }
