/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./mapped_range.hpp"
#include "./point_range.hpp"
#include "./polygon.hpp"
#include "./unit_vector_range.hpp"
namespace tf {
namespace implementation {
template <typename Range0> struct polygon_range_policy {
  Range0 points;
  template <typename Range> auto operator()(Range &&ids) const {
    return tf::make_polygon(ids, points);
  }

  template <typename Iterator0, typename Iterator1>
  auto operator()(std::pair<Iterator0, Iterator1> iters) const {
    return tf::make_polygon(*iters.first, points, *iters.second);
  }
};

template <typename Range0, typename Range1, typename Range2>
auto make_polygon_range(Range0 &&faces, Range1 &&points, Range2 &&normals) {
  auto r0 = tf::make_range(faces);
  auto r1 = tf::make_range(points);
  auto r2 = tf::make_range(normals);
  auto begins = std::make_pair(r0.begin(), r2.begin());
  auto ends = std::make_pair(r0.end(), r2.end());
  auto it =
      iter::make_iter_mapped(begins, polygon_range_policy<decltype(r1)>{r1});
  auto end =
      iter::make_iter_mapped(ends, polygon_range_policy<decltype(r1)>{r1});
  return tf::make_range(it, end);
}

} // namespace implementation

template <typename Range0, typename Range1, typename...>
struct polygon_range
    : decltype(tf::make_mapped_range(
          std::declval<Range0>(),
          std::declval<implementation::polygon_range_policy<Range1>>())) {
private:
  using base_t = decltype(tf::make_mapped_range(
      std::declval<Range0>(),
      std::declval<implementation::polygon_range_policy<Range1>>()));

public:
  polygon_range(const Range0 &faces, const Range1 &points)
      : base_t{tf::make_mapped_range(
            faces, implementation::polygon_range_policy<Range1>{points})} {}

  auto faces() const {
    return tf::make_range(base_t::begin().base_iter(), base_t::size());
  }

  auto points() const {
    return tf::make_point_range(base_t::begin().dereference_policy().points);
  }
};

template <typename Range0, typename Range1, typename Range2>
struct polygon_range<Range0, Range1, Range2>
    : decltype(implementation::make_polygon_range(std::declval<Range0>(),
                                                  std::declval<Range1>(),
                                                  std::declval<Range2>())) {
private:
  using base_t = decltype(implementation::make_polygon_range(
      std::declval<Range0>(), std::declval<Range1>(), std::declval<Range2>()));

public:
  polygon_range(const Range0 &faces, const Range1 &points,
                const Range2 &normals)
      : base_t{implementation::make_polygon_range(faces, points, normals)} {}

  auto faces() const {
    return tf::make_range(base_t::begin().base_iter().base_iter().first,
                          base_t::size());
  }

  auto points() const {
    return tf::make_point_range(
        base_t::begin().base_iter().dereference_policy().range);
  }

  auto normals() const {
    return tf::make_unit_vector_range(tf::make_range(
        base_t::begin().base_iter().base_iter().second, base_t::size()));
  }
};

/// @ingroup ranges
/// @brief Creates a range of polygons from indexed face and point data.
///
/// This utility constructs a range of @ref tf::polygon objects using a
/// block-indirect representation: each element in the `faces` range defines a
/// subrange of point indices, which are used to extract corresponding
/// points from the `points` range.
///
/// Each polygon is thus formed by mapping a contiguous block of indices
/// to actual point coordinates.
///
/// @tparam Range0 A range of integral indices grouped into blocks (e.g.,
/// faces).
/// @tparam Range1 A range of point data (e.g., `std::vector<vec2>`).
/// @param faces A range defining groups of indices, where each group forms a
/// polygon.
/// @param points A range of points from which polygons are constructed.
/// @return A @ref tf::range of @ref tf::polygon objects.
///
/// @note Internally uses @ref tf::make_block_indirect_range and
/// @ref tf::make_mapped_range with @ref
/// tf::implementation::polygon_range_policy.
/// @note This function returns a view — it does not copy the underlying data.
///
/// @code
/// tf::buffer<std::size_t> faces{0, 3, 6};
/// tf::buffer<std::size_t> ids{0, 1, 2, 2, 3, 0};
/// tf::buffer<vec2> points{{0,0}, {1,0}, {1,1}, {0,1}};
/// for (auto poly : make_polygon_range(make_offset_block_range(faces, ids),
/// points)) {
///   draw(poly);
/// }
/// @endcode
template <typename Range0, typename Range1>
auto make_polygon_range(Range0 &&faces, Range1 &&points) {
  auto r0 = tf::make_range(faces);
  auto r1 = tf::make_point_range(points);
  return polygon_range<decltype(r0), decltype(r1)>{r0, r1};
}

template <typename Range0, typename Range1, typename Range2>
auto make_polygon_range(Range0 &&faces, Range1 &&points, Range2 &&normals) {
  auto r0 = tf::make_range(faces);
  auto r1 = tf::make_point_range(points);
  auto r2 = tf::make_unit_vector_range(normals);
  return polygon_range<decltype(r0), decltype(r1), decltype(r2)>{r0, r1, r2};
}
} // namespace tf
