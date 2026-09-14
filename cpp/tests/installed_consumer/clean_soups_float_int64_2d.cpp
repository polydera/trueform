#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/clean/async/soup.hpp>
#include <trueform/cpp/clean/soup.hpp>

#include <cstdint>
#include <future>
#include <iostream>
#include <type_traits>

int main() {
  using namespace trueform_installed_test;
  using segment_result =
      tf::cpp::basic_cleaned_polygon_soup_result<std::int64_t, float, 2, 2>;
  using triangle_result =
      tf::cpp::basic_cleaned_polygon_soup_result<std::int64_t, float, 2, 3>;

  auto segments = make_array<float>(
      {0.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 1.0F}, {2, 2, 2});
  auto triangles = make_array<float>(
      {0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 2.0F, 0.0F, 1.0F, 1.0F},
      {2, 3, 2});

  const auto cleaned_segments =
      tf::cpp::cleaned_polygon_soup<std::int64_t, float, 2, 2>(segments);
  const auto cleaned_triangles =
      tf::cpp::cleaned_polygon_soup<std::int64_t, float, 2, 3>(triangles);
  auto pending_segments =
      tf::cpp::async::cleaned_polygon_soup<std::int64_t, float, 2, 2>(segments);
  auto pending_triangles =
      tf::cpp::async::cleaned_polygon_soup<std::int64_t, float, 2, 3>(
          triangles);

  static_assert(
      std::is_same_v<std::decay_t<decltype(cleaned_segments)>, segment_result>);
  static_assert(std::is_same_v<std::decay_t<decltype(cleaned_triangles)>,
                               triangle_result>);
  static_assert(
      std::is_same_v<decltype(pending_segments), std::future<segment_result>>);
  static_assert(std::is_same_v<decltype(pending_triangles),
                               std::future<triangle_result>>);

  segments.destroy();
  triangles.destroy();
  const auto async_segments = pending_segments.get();
  const auto async_triangles = pending_triangles.get();
  const auto &segment_edges =
      cleaned_segments.edges_buffer().data_buffer();
  const auto ok = cleaned_segments.size() == 2 &&
                  cleaned_segments.points_buffer().size() == 3 &&
                  segment_edges[0] == std::int64_t{0} &&
                  segment_edges[1] == std::int64_t{1} &&
                  segment_edges[2] == std::int64_t{1} &&
                  segment_edges[3] == std::int64_t{2} &&
                  cleaned_triangles.size() == 2 &&
                  cleaned_triangles.points_buffer().size() == 5 &&
                  async_segments.size() == 2 &&
                  async_segments.points_buffer().size() == 3 &&
                  async_triangles.size() == 2 &&
                  async_triangles.points_buffer().size() == 5;
  if (!ok)
    std::cerr << "float/int64/2D segment or triangle soup cleaning mismatch\n";
  return ok ? 0 : 1;
}
