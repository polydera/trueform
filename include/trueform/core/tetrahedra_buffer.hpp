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
#include "./algorithm/parallel_copy.hpp"
#include "./base/tetrahedra.hpp"
#include "./blocked_buffer.hpp"
#include "./coordinate_type.hpp"
#include "./points.hpp"
#include "./points_buffer.hpp"
#include "./tetrahedra.hpp"
#include "./tetras.hpp"
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup core_buffers
/// @brief Owns flat four-index blocks and point coordinates; tetrahedra()
/// borrows them.
template <typename Index, typename RealT> class tetrahedra_buffer {
public:
  using iterator = decltype(core::make_tetrahedron_range_iter(
      std::declval<tf::blocked_buffer<Index, 4> &>().begin(),
      std::declval<tf::points_buffer<RealT, 3> &>()));
  using const_iterator = decltype(core::make_tetrahedron_range_iter(
      std::declval<const tf::blocked_buffer<Index, 4> &>().begin(),
      std::declval<const tf::points_buffer<RealT, 3> &>()));
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using const_reference =
      typename std::iterator_traits<const_iterator>::reference;
  using pointer = typename std::iterator_traits<iterator>::pointer;
  using size_type = std::size_t;

  auto begin() const -> const_iterator {
    return core::make_tetrahedron_range_iter(_tetras_buffer.begin(),
                                             points_buffer());
  }

  auto begin() -> iterator {
    return core::make_tetrahedron_range_iter(_tetras_buffer.begin(),
                                             points_buffer());
  }

  auto end() const -> const_iterator {
    return core::make_tetrahedron_range_iter(_tetras_buffer.end(),
                                             points_buffer());
  }

  auto end() -> iterator {
    return core::make_tetrahedron_range_iter(_tetras_buffer.end(),
                                             points_buffer());
  }

  auto size() const -> size_type { return _tetras_buffer.size(); }

  auto empty() const -> bool { return _tetras_buffer.size() == 0; }

  auto front() const -> const_reference { return *begin(); }

  auto front() -> reference { return *begin(); }

  auto tetrahedra() const { return tf::make_tetrahedra(tetras(), points()); }

  auto tetrahedra() { return tf::make_tetrahedra(tetras(), points()); }

  auto points() const { return tf::make_points(points_buffer()); }

  auto points() { return tf::make_points(points_buffer()); }

  auto tetras() const { return tf::make_tetras(tetras_buffer()); }

  auto tetras() { return tf::make_tetras(tetras_buffer()); }

  auto tetras_buffer() -> tf::blocked_buffer<Index, 4> & {
    return _tetras_buffer;
  }

  auto tetras_buffer() const -> const tf::blocked_buffer<Index, 4> & {
    return _tetras_buffer;
  }

  auto points_buffer() -> tf::points_buffer<RealT, 3> & {
    return _points_buffer;
  }

  auto points_buffer() const -> const tf::points_buffer<RealT, 3> & {
    return _points_buffer;
  }

  auto clear() {
    _points_buffer.clear();
    _tetras_buffer.clear();
  }

private:
  tf::blocked_buffer<Index, 4> _tetras_buffer;
  tf::points_buffer<RealT, 3> _points_buffer;
};
/// @ingroup core_buffers
/// @brief Transfer ownership of the connectivity and coordinate buffers.
template <typename Index, typename RealT>
auto make_tetrahedra_buffer(tf::blocked_buffer<Index, 4> &&tetras,
                            tf::points_buffer<RealT, 3> &&points) {
  tf::tetrahedra_buffer<Index, RealT> out;
  out.tetras_buffer() = std::move(tetras);
  out.points_buffer() = std::move(points);
  return out;
}

/// @ingroup core_buffers
/// @brief Copy indexed connectivity and all points into independent storage.
template <typename Policy>
auto make_tetrahedra_buffer(const tf::tetrahedra<Policy> &tetrahedra) {
  using Index = std::decay_t<decltype(tetrahedra.tetras()[0][0])>;
  using RealT = tf::coordinate_type<Policy>;
  tf::tetrahedra_buffer<Index, RealT> out;
  out.points_buffer().allocate(tetrahedra.points().size());
  tf::parallel_copy(tetrahedra.points(), out.points());
  out.tetras_buffer().allocate(tetrahedra.size());
  tf::parallel_copy(tetrahedra.tetras(), out.tetras());
  return out;
}

} // namespace tf
