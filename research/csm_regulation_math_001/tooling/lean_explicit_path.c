/*
 * Environment portability launcher for the unmodified Lean 4.19 shared library.
 *
 * Usage: lean-explicit-path /absolute/toolchain/bin/lean [Lean arguments...]
 *
 * This environment's numeric process IDs do not match its /proc namespace.
 * Lean 4.19's application-path discovery therefore fails before elaboration.
 * This launcher supplies the already known application path explicitly. It
 * performs no /proc access, changes no filesystem permissions or sandbox
 * settings, and does not replace any elaborator, kernel, or proof operation.
 * The actual entry point remains the distribution's lean_main.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static const char *explicit_app_path = NULL;

/* Lean binds its own runtime symbols locally. Adapt just its libc application-
 * path request instead. Return supplied data; never read a different proc entry.
 * Every other readlink request uses libc with its usual access checks. */
ssize_t readlink(const char *path, char *buf, size_t bufsiz) {
    char self_request[64];
    snprintf(self_request, sizeof(self_request), "/proc/%ld/exe", (long)getpid());
    if (explicit_app_path != NULL && strcmp(path, self_request) == 0) {
        size_t size = strlen(explicit_app_path);
        if (size > bufsiz) size = bufsiz;
        memcpy(buf, explicit_app_path, size);
        return (ssize_t)size;
    }
    ssize_t (*libc_readlink)(const char *, char *, size_t) =
        (ssize_t (*)(const char *, char *, size_t))dlsym(RTLD_NEXT, "readlink");
    return libc_readlink(path, buf, bufsiz);
}

extern int lean_main(int argc, char **argv);

int main(int argc, char **argv) {
    if (argc < 2 || argv[1][0] != '/') {
        fputs("usage: lean-explicit-path /absolute/toolchain/bin/lean [arguments...]\n", stderr);
        return 2;
    }
    explicit_app_path = argv[1];
    return lean_main(argc - 1, argv + 1);
}
