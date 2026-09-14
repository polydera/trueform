#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/arrangement/mesh_arrangements.hpp>
#include <trueform/cpp/arrangement/polygon_arrangements.hpp>

#include <cstdint>
#include <type_traits>
#include <vector>

int main() {
  using namespace trueform_installed_test;
  using real_type = double;
  using index_type = std::int64_t;
  // a range is homogeneous, so its element states the arity the operands are
  // read at
  constexpr auto mixed = tf::dynamic_size;
  using mesh_type = tf::cpp::mesh<index_type, real_type, 3, mixed>;

  const auto first = empty_mesh_3d<index_type, real_type, mixed>();
  const auto second = empty_mesh_3d<index_type, real_type, mixed>();
  const std::vector<mesh_type> forms{first.mesh(), second.mesh()};

  const auto mesh_result = tf::cpp::mesh_arrangements(forms);
  const auto mesh_curves = tf::cpp::mesh_arrangements_with_curves(forms);
  static_assert(
      std::is_same_v<
          std::decay_t<decltype(mesh_result)>,
          tf::cpp::mesh_arrangement_result<index_type, real_type, mixed>>);
  static_assert(std::is_same_v<std::decay_t<decltype(mesh_curves)>,
                               tf::cpp::mesh_arrangement_with_curves_result<
                                   index_type, real_type, mixed>>);

  const auto polygon = empty_mesh_3d<index_type, real_type, mixed>();
  const auto polygon_result = tf::cpp::polygon_arrangements(polygon.mesh());
  const auto polygon_curves =
      tf::cpp::polygon_arrangements_with_curves(polygon.mesh());
  static_assert(
      std::is_same_v<
          std::decay_t<decltype(polygon_result)>,
          tf::cpp::polygon_arrangement_result<index_type, real_type, mixed>>);
  static_assert(std::is_same_v<std::decay_t<decltype(polygon_curves)>,
                               tf::cpp::polygon_arrangement_with_curves_result<
                                   index_type, real_type, mixed>>);

  const auto ok = mesh_result.mesh.size() == 0 &&
                  mesh_curves.curves.size() == 0 &&
                  polygon_result.face_labels.length() == 0 &&
                  polygon_curves.curves.size() == 0;
  return ok ? 0 : 1;
}
