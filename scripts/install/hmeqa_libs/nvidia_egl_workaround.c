/* nvidia_egl_workaround.c
 *
 * Workaround for an NVIDIA driver-570 + Magnum/Corrade interaction that
 * crashes habitat-sim 0.3.x at Simulator() construction time on headless
 * (EGL) GPUs.
 *
 * Symptom
 * -------
 *   SIGSEGV in __strlen_avx2 inside Corrade::Containers::BasicStringView,
 *   reached from Magnum::Platform::WindowlessEglContext at
 *   .../Magnum/Platform/WindowlessEglApplication.cpp:492 :
 *
 *       const Containers::StringView vendorString =
 *           reinterpret_cast<const char*>(glGetString(GL_VENDOR));
 *
 * Root cause
 * ----------
 *   On NVIDIA driver 570.x with the headless EGL path, eglMakeCurrent
 *   appears to succeed but does not fully bind a usable GL context. The
 *   subsequent glGetString(GL_VENDOR) call returns a *bogus* pointer
 *   value (e.g. 0xffffffffffffff68 — a small negative integer cast to
 *   pointer) instead of a NULL or a valid string. Corrade's StringView
 *   has a NULL-fast-path (handles glGetString returning NULL gracefully),
 *   but it has no way to detect a non-NULL bogus pointer, so it strlen's
 *   into invalid memory.
 *
 * Strategy
 * --------
 *   Intercept glGetString via LD_PRELOAD. If the underlying driver
 *   returns a pointer that is clearly outside the user-space address
 *   range (NULL is OK; tiny ints like 0x4 or huge negatives like
 *   0xff..ff are not real strings), substitute NULL — which triggers
 *   Corrade's NULL fast-path and lets Magnum continue without the
 *   vendor-specific workaround. Valid strings are forwarded untouched.
 *
 *   This is safe to install in an LD_PRELOAD chain wholesale: legitimate
 *   GL drivers always return either NULL or a pointer inside their
 *   .rodata segment, both of which we forward unchanged.
 *
 * Build
 * -----
 *   gcc -shared -fPIC -O2 -o nvidia_egl_workaround.so \
 *       nvidia_egl_workaround.c -ldl
 *
 * Use
 * ---
 *   export LD_PRELOAD=/path/to/nvidia_egl_workaround.so
 *
 * Verification
 * ------------
 *   See scripts/install/hmeqa_libs/test_workaround.py — boots a real
 *   HM3D scene and renders a 480×640 RGB+depth observation.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>

typedef unsigned int GLenum;
typedef unsigned char GLubyte;

/* User-space address-range bounds on x86_64 Linux. Real strings live well
 * inside this band (typical .rodata addresses are around 0x7f00_0000_0000
 * for shared libs and slightly lower for the main exe). Anything outside
 * is the driver-570 bug. */
#define MIN_VALID_PTR ((uintptr_t)0x10000)
#define MAX_VALID_PTR ((uintptr_t)0x800000000000)

const GLubyte* glGetString(GLenum name) {
    static const GLubyte* (*real)(GLenum) = NULL;
    static int resolved = 0;
    if (!resolved) {
        real = (const GLubyte*(*)(GLenum))dlsym(RTLD_NEXT, "glGetString");
        resolved = 1;
    }
    if (!real) {
        /* No real glGetString reachable past us in the load order
         * (this can happen with libGLdispatch's TLS-stub design).
         * Return NULL — Magnum/Corrade handle that path correctly. */
        return NULL;
    }

    const GLubyte* r = real(name);
    if (!r) return NULL;

    uintptr_t p = (uintptr_t)r;
    if (p < MIN_VALID_PTR || p >= MAX_VALID_PTR) {
        /* Bogus pointer from buggy driver — forge to NULL so the caller's
         * NULL-handling kicks in instead of strlen'ing into garbage. */
        return NULL;
    }
    return r;
}
