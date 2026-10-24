/* hipmem-watch-02: diagnostic LD_PRELOAD, passes every call through. On the process's first hipMemMap it also reserves a second
   address range B and maps the same handle there (read-write), so a test can read that memory through two addresses.
   hipmem_watch_a() / hipmem_watch_b() / hipmem_watch_size() return the first mapping, the second mapping and its size. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stddef.h>
typedef int hipError_t; typedef void *hipHandle;
typedef struct { int type; int id; } Loc; typedef struct { Loc location; int flags; } Access;
static hipError_t (*r_map)(void *, size_t, size_t, hipHandle, unsigned long long);
static hipError_t (*r_reserve)(void **, size_t, size_t, void *, unsigned long long);
static hipError_t (*r_access)(void *, size_t, const Access *, size_t);
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static void *wa, *wb; static size_t ws;
void *hipmem_watch_a(void) { return wa; } void *hipmem_watch_b(void) { return wb; } size_t hipmem_watch_size(void) { return ws; }
static void *get(const char *n) { void *f = dlvsym(RTLD_NEXT, n, "hip_5.1"); return f ? f : dlsym(RTLD_NEXT, n); }
hipError_t hipMemMap(void *p, size_t s, size_t o, hipHandle h, unsigned long long f) {
  pthread_mutex_lock(&mu);
  if (!r_map) { r_map = get("hipMemMap"); r_reserve = get("hipMemAddressReserve"); r_access = get("hipMemSetAccess"); }
  hipError_t e = r_map(p, s, o, h, f);
  if (e == 0 && !wa) {
    void *b = NULL; Access acc = {{1, 0}, 3};
    hipError_t e1 = r_reserve(&b, s, 0, NULL, 0), e2 = e1 ? -1 : r_map(b, s, 0, h, 0), e3 = (e1 || e2) ? -1 : r_access(b, s, &acc, 1);
    wa = p; ws = s; if (!e1 && !e2 && !e3) wb = b;
    fprintf(stderr, "[hipmem-watch-02] first map %p +%zu; second mapping of the same handle at %p (reserve %d, map %d, access %d)\n", p, s, b, e1, e2, e3);
  }
  pthread_mutex_unlock(&mu); return e;
}
