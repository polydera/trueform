#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/geometry/normals.hpp>
#include <trueform/cpp/geometry/point_normals.hpp>

#include <cstdint>

int main() {
  using real_type = double;
  using index_type = std::int64_t;
  using namespace trueform_installed_test;

  const consumer::owned_mesh<index_type, real_type, 3, tf::dynamic_size> value{
      polygons_of<index_type, real_type, 3>(
          {0, 3, 7}, {0, 1, 2, 1, 3, 4, 2},
          {0, 0, 0, 1, 0, 0, 0.5, 1, 0, 2, 0, 0, 1.5, 1, 0})};

  const auto face_values =
      tf::cpp::normals<index_type, real_type, 3>(value.mesh());
  const auto point_values =
      tf::cpp::point_normals<index_type, real_type, 3>(value.mesh());
  if (face_values.raw_shape() != tf::small_vector<int, 3>{2, 3} ||
      point_values.raw_shape() != tf::small_vector<int, 3>{5, 3})
    return 1;
  for (const auto &normals : {face_values, point_values})
    for (std::size_t index = 0; index < normals.length(); ++index) {
      const auto expected = index % 3 == 2 ? real_type{1} : real_type{0};
      if (normals[index] != expected)
        return 2;
    }
  return 0;
}
