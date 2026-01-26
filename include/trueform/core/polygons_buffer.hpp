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
#include "./base/polygons.hpp"
#include "./blocked_buffer.hpp"
#include "./faces.hpp"
#include "./offset_block_buffer.hpp"
#include "./points.hpp"
#include "./points_buffer.hpp"
#include "./polygons.hpp"
namespace tf {

/// @ingroup core_buffers
/// @brief An owning buffer of polygons with fixed vertex count.
///
/// Stores face indices and point coordinates separately. The `Ngon`
/// parameter specifies the fixed number of vertices per polygon.
/// Use `polygons()` to obtain a @ref tf::polygons range.
///
/// For variable-sized polygons, use the @ref tf::dynamic_size specialization.
///
/// @tparam Index The index type for face connectivity.
/// @tparam RealT The coordinate scalar type.
/// @tparam Dims The number of dimensions.
/// @tparam Ngon The number of vertices per polygon (or @ref tf::dynamic_size).
template <typename Index, typename RealT, std::size_t Dims, std::size_t Ngon>
class polygons_buffer {
public:
  using iterator = decltype(core::make_polygon_range_iter(
      std::declval<tf::blocked_buffer<Index, Ngon> &>().begin(),
      std::declval<tf::points_buffer<RealT, Dims> &>()));
  using const_iterator = decltype(core::make_polygon_range_iter(
      std::declval<const tf::blocked_buffer<Index, Ngon> &>().begin(),
      std::declval<const tf::points_buffer<RealT, Dims> &>()));
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using const_reference =
      typename std::iterator_traits<const_iterator>::reference;
  using pointer = typename std::iterator_traits<iterator>::pointer;
  using size_type = std::size_t;

  auto begin() const -> const_iterator {
    return core::make_polygon_range_iter(_faces_buffer.begin(),
                                         points_buffer());
  }

  auto begin() -> iterator {
    return core::make_polygon_range_iter(_faces_buffer.begin(),
                                         points_buffer());
  }

  auto end() const -> const_iterator {
    return core::make_polygon_range_iter(_faces_buffer.end(), points_buffer());
  }

  auto end() -> iterator {
    return core::make_polygon_range_iter(_faces_buffer.end(), points_buffer());
  }

  auto size() const -> size_type { return _faces_buffer.size(); }

  auto empty() const -> bool { return _faces_buffer.size() == 0; }

  auto front() const -> const_reference { return *begin(); }

  auto front() -> reference { return *begin(); }

  auto polygons() const { return tf::make_polygons(faces(), points()); }

  auto polygons() { return tf::make_polygons(faces(), points()); }

  auto points() const { return tf::make_points(points_buffer()); }

  auto points() { return tf::make_points(points_buffer()); }

  auto faces() const { return tf::make_faces(faces_buffer()); }

  auto faces() { return tf::make_faces(faces_buffer()); }

  auto faces_buffer() -> tf::blocked_buffer<Index, Ngon> & {
    return _faces_buffer;
  }

  auto faces_buffer() const -> const tf::blocked_buffer<Index, Ngon> & {
    return _faces_buffer;
  }

  auto points_buffer() -> tf::points_buffer<RealT, Dims> & {
    return _points_buffer;
  }

  auto points_buffer() const -> const tf::points_buffer<RealT, Dims> & {
    return _points_buffer;
  }

  auto clear() {
    _points_buffer.clear();
    _faces_buffer.clear();
  }

private:
  tf::blocked_buffer<Index, Ngon> _faces_buffer;
  tf::points_buffer<RealT, Dims> _points_buffer;
};

/// @ingroup core_buffers
/// @brief Specialization for polygons with variable vertex count.
///
/// Uses offset-based storage for variable-sized polygons.
template <typename Index, typename RealT, std::size_t Dims>
class polygons_buffer<Index, RealT, Dims, tf::dynamic_size> {
  using iterator = decltype(core::make_polygon_range_iter(
      std::declval<tf::offset_block_buffer<Index, Index> &>().begin(),
      std::declval<tf::points_buffer<RealT, Dims> &>()));
  using const_iterator = decltype(core::make_polygon_range_iter(
      std::declval<const tf::offset_block_buffer<Index, Index> &>().begin(),
      std::declval<const tf::points_buffer<RealT, Dims> &>()));
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using const_reference =
      typename std::iterator_traits<const_iterator>::reference;
  using pointer = typename std::iterator_traits<iterator>::pointer;
  using size_type = std::size_t;

public:
  auto begin() const -> const_iterator {
    return core::make_polygon_range_iter(_faces_buffer.begin(),
                                         points_buffer());
  }

  auto begin() -> iterator {
    return core::make_polygon_range_iter(_faces_buffer.begin(),
                                         points_buffer());
  }

  auto end() const -> const_iterator {
    return core::make_polygon_range_iter(_faces_buffer.end(), points_buffer());
  }

  auto end() -> iterator {
    return core::make_polygon_range_iter(_faces_buffer.end(), points_buffer());
  }

  auto size() const -> size_type { return _faces_buffer.size(); }

  auto empty() const -> bool { return _faces_buffer.size() == 0; }

  auto front() const -> const_reference { return *begin(); }

  auto front() -> reference { return *begin(); }

  auto polygons() const { return tf::make_polygons(faces(), points()); }

  auto polygons() { return tf::make_polygons(faces(), points()); }

  auto points() const { return tf::make_points(points_buffer()); }

  auto points() { return tf::make_points(points_buffer()); }

  auto faces() const { return tf::make_faces(faces_buffer()); }

  auto faces() { return tf::make_faces(faces_buffer()); }

  auto faces_buffer() -> tf::offset_block_buffer<Index, Index> & {
    return _faces_buffer;
  }

  auto faces_buffer() const -> const tf::offset_block_buffer<Index, Index> & {
    return _faces_buffer;
  }

  auto points_buffer() -> tf::points_buffer<RealT, Dims> & {
    return _points_buffer;
  }

  auto points_buffer() const -> const tf::points_buffer<RealT, Dims> & {
    return _points_buffer;
  }

  auto clear() {
    _points_buffer.clear();
    _faces_buffer.clear();
  }

private:
  tf::offset_block_buffer<Index, Index> _faces_buffer;
  tf::points_buffer<RealT, Dims> _points_buffer;
};

/// @ingroup core_buffers
/// @brief Create a polygons buffer from variable-sized faces and points.
template <typename Index, typename RealT, std::size_t Dims>
auto make_polygons_buffer(tf::offset_block_buffer<Index, Index> &&faces,
                          tf::points_buffer<RealT, Dims> &&points) {
  tf::polygons_buffer<Index, RealT, Dims, tf::dynamic_size> out;
  out.faces_buffer() = std::move(faces);
  out.points_buffer() = std::move(points);
  return out;
}

/// @ingroup core_buffers
/// @brief Create a polygons buffer from fixed-size faces and points.
template <typename Index, std::size_t N, typename RealT, std::size_t Dims>
auto make_polygons_buffer(tf::blocked_buffer<Index, N> &&faces,
                          tf::points_buffer<RealT, Dims> &&points) {
  tf::polygons_buffer<Index, RealT, Dims, N> out;
  out.faces_buffer() = std::move(faces);
  out.points_buffer() = std::move(points);
  return out;
}
} // namespace tf
