#include <trueform/core/polygons_buffer.hpp>
#include <trueform/cpp/core/cache.hpp>
#include <trueform/cpp/core/mesh.hpp>
#include <trueform/cpp/csg/outer_shell.hpp>

#include <cstdint>

int main() {
  const tf::polygons_buffer<std::int64_t, long double, 3, 3> storage;
  tf::cpp::cache<std::int64_t, long double, 3, 3> cache;
  const tf::cpp::mesh<std::int64_t, long double, 3, 3> unsupported{
      storage.faces(), storage.points(), cache};
  auto shell = tf::cpp::outer_shell<std::int64_t, long double>(unsupported);
  return static_cast<int>(shell.size());
}
