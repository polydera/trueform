#include <emscripten/bind.h>
int add(int a, int b) {
    return a + b;
}

EMSCRIPTEN_BINDINGS(my_module) {
  emscripten::function("add", &add);
}