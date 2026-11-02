#include "livevoice.h"

#include <cstdio>
#include <cstring>

int main() {
    const char* v = lv_version();
    if (!v || !std::strlen(v)) {
        std::fprintf(stderr, "FAIL: empty version\n");
        return 1;
    }
    std::printf("%s\n", v);
    return 0;
}
