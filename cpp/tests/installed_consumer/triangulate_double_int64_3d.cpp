#include "triangulation_axis_shard_test.hpp"

#include <cstdint>

int main() {
  return trueform_installed_test::check_triangulation_axis_shard<std::int64_t,
                                                                 double, 3>()
             ? 0
             : 1;
}
