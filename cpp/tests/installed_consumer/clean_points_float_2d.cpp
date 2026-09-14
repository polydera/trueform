#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/clean/points.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>

namespace {

template <typename T>
auto equal_arrays(const tf::cpp::nd_array<T> &first,
                  const tf::cpp::nd_array<T> &second) -> bool {
  return first.raw_shape() == second.raw_shape() &&
         std::equal(first.begin(), first.end(), second.begin());
}

} // namespace

int main() {
  using namespace trueform_installed_test;
  auto input = make_array<float>({0, 0, 1, 0, 0, 0, 2, 1}, {4, 2});
  const consumer::owned_point_cloud<float, 2> owned{
      points_of<float, 2>({0, 0, 1, 0, 0, 0, 2, 1})};
  const auto cloud = owned.point_cloud();

  const auto array_points = tf::cpp::cleaned_points<float, 2>(input);
  const auto array_result = tf::cpp::cleaned_points_with_map<float, 2>(input);
  const auto cloud_points = tf::cpp::cleaned_points<float, 2>(cloud);
  const auto cloud_result = tf::cpp::cleaned_points_with_map<float, 2>(cloud);

  const auto ok =
      array_points.raw_shape() == tf::small_vector<int, 3>{3, 2} &&
      equal_arrays(array_points, array_result.points) &&
      equal_arrays(array_points, cloud_points) &&
      equal_arrays(array_points, cloud_result.points) &&
      array_result.point_map.f.length() == 4 &&
      array_result.point_map.kept_ids.length() == 3 &&
      array_result.point_map.f[0] == array_result.point_map.f[2] &&
      array_result.point_map.f[0] != array_result.point_map.f[1] &&
      array_result.point_map.f[1] != array_result.point_map.f[3] &&
      equal_arrays(array_result.point_map.f, cloud_result.point_map.f) &&
      equal_arrays(array_result.point_map.kept_ids,
                   cloud_result.point_map.kept_ids) &&
      array_points.raw_data() != input.raw_data();
  if (!ok)
    std::cerr << "float/2D point cleaning mismatch\n";
  return ok ? 0 : 1;
}
