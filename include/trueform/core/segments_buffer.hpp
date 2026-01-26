/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/
#pragma once
#include "./base/segments.hpp"
#include "./blocked_buffer.hpp"
#include "./edges.hpp"
#include "./points.hpp"
#include "./points_buffer.hpp"
#include "./segments.hpp"
namespace tf {

/// @ingroup core_buffers
/// @brief An owning buffer of line segments.
///
/// Stores edge indices and point coordinates separately. Use `segments()`
/// to obtain a @ref tf::segments range, `edges()` for the edge indices,
/// or `points()` for the point data.
///
/// @tparam Index The index type for edge connectivity.
/// @tparam RealT The coordinate scalar type.
/// @tparam Dims The number of dimensions.
template <typename Index, typename RealT, std::size_t Dims>
class segments_buffer {
public:
  using iterator = decltype(core::make_segment_range_iter(
      std::declval<tf::blocked_buffer<Index, 2> &>().begin(),
      std::declval<tf::points_buffer<RealT, Dims> &>()));
  using const_iterator = decltype(core::make_segment_range_iter(
      std::declval<const tf::blocked_buffer<Index, 2> &>().begin(),
      std::declval<const tf::points_buffer<RealT, Dims> &>()));
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using const_reference =
      typename std::iterator_traits<const_iterator>::reference;
  using pointer = typename std::iterator_traits<iterator>::pointer;
  using size_type = std::size_t;

  auto begin() const -> const_iterator {
    return core::make_segment_range_iter(_edges_buffer.begin(),
                                         points_buffer());
  }

  auto begin() -> iterator {
    return core::make_segment_range_iter(_edges_buffer.begin(),
                                         points_buffer());
  }

  auto end() const -> const_iterator {
    return core::make_segment_range_iter(_edges_buffer.end(), points_buffer());
  }

  auto end() -> iterator {
    return core::make_segment_range_iter(_edges_buffer.end(), points_buffer());
  }

  auto size() const -> size_type { return _edges_buffer.size(); }

  auto empty() const -> bool { return _edges_buffer.size() == 0; }

  auto front() const -> const_reference { return *begin(); }

  auto front() -> reference { return *begin(); }

  auto segments() const { return tf::make_segments(edges(), points()); }

  auto segments() { return tf::make_segments(edges(), points()); }

  auto points() const { return tf::make_points(points_buffer()); }

  auto points() { return tf::make_points(points_buffer()); }

  auto edges() const { return tf::make_edges(edges_buffer()); }

  auto edges() { return tf::make_edges(edges_buffer()); }

  auto edges_buffer() -> tf::blocked_buffer<Index, 2> & {
    return _edges_buffer;
  }

  auto edges_buffer() const -> const tf::blocked_buffer<Index, 2> & {
    return _edges_buffer;
  }

  auto points_buffer() -> tf::points_buffer<RealT, Dims> & {
    return _points_buffer;
  }

  auto points_buffer() const -> const tf::points_buffer<RealT, Dims> & {
    return _points_buffer;
  }

  auto clear() {
    _points_buffer.clear();
    _edges_buffer.clear();
  }

private:
  tf::blocked_buffer<Index, 2> _edges_buffer;
  tf::points_buffer<RealT, Dims> _points_buffer;
};
} // namespace tf
