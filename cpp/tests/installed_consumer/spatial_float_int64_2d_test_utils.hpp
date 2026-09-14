#pragma once

#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/core/nd_array.hpp>
#include <trueform/cpp/spatial/primitive.hpp>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <utility>

namespace spatial_float_int64_2d_test {

template <typename T>
auto make_array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> storage;
  storage.allocate(values.size());
  std::copy(values.begin(), values.end(), storage.begin());
  return tf::cpp::nd_array<T>::from_buffer(std::move(storage),
                                           std::move(shape));
}

inline auto triangle_mesh() -> consumer::owned_mesh<std::int64_t, float, 2> {
  return {trueform_installed_test::polygons_of<std::int64_t, float, 2>(
      {0, 1, 2}, {0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F})};
}

inline auto point(float x = 0.25F, float y = 0.25F)
    -> tf::cpp::primitive<float, 2> {
  return {tf::cpp::primitive_kind::point, make_array<float>({x, y}, {2})};
}

inline auto ray() -> tf::cpp::primitive<float, 2> {
  return {tf::cpp::primitive_kind::ray,
          make_array<float>({-1.0F, 0.25F, 1.0F, 0.0F}, {2, 2})};
}

} // namespace spatial_float_int64_2d_test
