#include "topology_analysis_axis_test.hpp"

#include <cstdint>

int main() {
  return trueform_installed_test::check_topology_analysis_axis<std::int64_t,
                                                               float, 2>()
             ? 0
             : 1;
}
