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
#include "./algorithm/parallel_copy_blocked.hpp"
#include "./base/curves.hpp"
#include "./coordinate_type.hpp"
#include "./curves.hpp"
#include "./offset_block_buffer.hpp"
#include "./paths.hpp"
#include "./points.hpp"
#include "./points_buffer.hpp"
#include "./views/enumerate.hpp"
namespace tf {

/// @ingroup core_buffers
/// @brief An owning buffer of polyline curves.
///
/// Stores path indices and point coordinates separately using offset-based
/// storage for variable-length curves. Use `curves()` to obtain a
/// @ref tf::curves range.
///
/// @tparam Index The index type for path connectivity.
/// @tparam RealT The coordinate scalar type.
/// @tparam Dims The number of dimensions.
template <typename Index, typename RealT, std::size_t Dims>
class curves_buffer {
  using iterator = decltype(core::make_curve_range_iter(
      std::declval<tf::offset_block_buffer<Index, Index> &>().begin(),
      std::declval<tf::points_buffer<RealT, Dims> &>()));
  using const_iterator = decltype(core::make_curve_range_iter(
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
    return core::make_curve_range_iter(_paths_buffer.begin(), points_buffer());
  }

  auto begin() -> iterator {
    return core::make_curve_range_iter(_paths_buffer.begin(), points_buffer());
  }

  auto end() const -> const_iterator {
    return core::make_curve_range_iter(_paths_buffer.end(), points_buffer());
  }

  auto end() -> iterator {
    return core::make_curve_range_iter(_paths_buffer.end(), points_buffer());
  }

  auto size() const -> size_type { return _paths_buffer.size(); }

  auto empty() const -> bool { return _paths_buffer.size() == 0; }

  auto front() const -> const_reference { return *begin(); }

  auto front() -> reference { return *begin(); }

  auto curves() const { return tf::make_curves(paths(), points()); }

  auto curves() { return tf::make_curves(paths(), points()); }

  auto points() const { return tf::make_points(points_buffer()); }

  auto points() { return tf::make_points(points_buffer()); }

  auto paths() const { return tf::make_paths(paths_buffer()); }

  auto paths() { return tf::make_paths(paths_buffer()); }

  auto paths_buffer() -> tf::offset_block_buffer<Index, Index> & {
    return _paths_buffer;
  }

  auto paths_buffer() const -> const tf::offset_block_buffer<Index, Index> & {
    return _paths_buffer;
  }

  auto points_buffer() -> tf::points_buffer<RealT, Dims> & {
    return _points_buffer;
  }

  auto points_buffer() const -> const tf::points_buffer<RealT, Dims> & {
    return _points_buffer;
  }

  auto clear() {
    _points_buffer.clear();
    _paths_buffer.clear();
  }

private:
  tf::offset_block_buffer<Index, Index> _paths_buffer;
  tf::points_buffer<RealT, Dims> _points_buffer;
};
/// @ingroup core_buffers
/// @brief Create a curves buffer from paths and points buffers.
template <typename Index, typename RealT, std::size_t Dims>
auto make_curves_buffer(tf::offset_block_buffer<Index, Index> &&paths,
                        tf::points_buffer<RealT, Dims> &&points) {
  tf::curves_buffer<Index, RealT, Dims> out;
  out.paths_buffer() = std::move(paths);
  out.points_buffer() = std::move(points);
  return out;
}

/// @ingroup core_buffers
/// @brief Create a curves buffer from a curves view.
template <typename Policy>
auto make_curves_buffer(const tf::curves<Policy> &curves) {
  using Index = std::decay_t<decltype(curves.paths()[0][0])>;
  using RealT = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  tf::curves_buffer<Index, RealT, Dims> out;
  if (!curves.size())
    return out;
  out.points_buffer().allocate(curves.points().size());
  tf::parallel_copy(curves.points(), out.points());
  auto &offsets = out.paths_buffer().offsets_buffer();
  offsets.allocate(curves.size() + 1);
  offsets[0] = 0;
  for (auto [i, p] : tf::enumerate(curves.paths()))
    offsets[i + 1] = offsets[i] + static_cast<Index>(p.size());
  out.paths_buffer().data_buffer().allocate(offsets.back());
  tf::parallel_copy_blocked(curves.paths(), out.paths());
  return out;
}

} // namespace tf
