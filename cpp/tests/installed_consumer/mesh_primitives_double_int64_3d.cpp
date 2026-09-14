#include <trueform/core/polygons_buffer.hpp>
#include <trueform/cpp/geometry/make_box_mesh.hpp>
#include <trueform/cpp/geometry/make_cylinder_mesh.hpp>
#include <trueform/cpp/geometry/make_plane_mesh.hpp>
#include <trueform/cpp/geometry/make_sphere_mesh.hpp>

#include <cstdint>
#include <type_traits>

int main() {
  using storage_type = tf::polygons_buffer<std::int64_t, double, 3, 3>;

  const auto sphere =
      tf::cpp::make_sphere_mesh<std::int64_t, double>(1.0, 2, 3);
  const auto cylinder =
      tf::cpp::make_cylinder_mesh<std::int64_t, double>(1.0, 2.0, 3);
  const auto box = tf::cpp::make_box_mesh<std::int64_t, double>(2.0, 4.0, 6.0);
  const auto subdivided_box =
      tf::cpp::make_box_mesh<std::int64_t, double>(2.0, 4.0, 6.0, 1, 1, 1);
  const auto plane =
      tf::cpp::make_plane_mesh<std::int64_t, double>(2.0, 4.0, 1, 1);

  static_assert(std::is_same_v<std::decay_t<decltype(sphere)>, storage_type>);
  return sphere.size() > 0 && cylinder.size() > 0 && box.size() > 0 &&
                 subdivided_box.size() > 0 && plane.size() > 0
             ? 0
             : 1;
}
