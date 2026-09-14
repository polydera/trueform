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
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/form.hpp"
#include "../core/point.hpp"
#include "../core/point_like.hpp"
#include "../core/range.hpp"
#include "./impl/grid.hpp"
#include "./impl/grid_sample_count.hpp"
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup volume
/// @brief A dense, axis-aligned scalar field on a regular grid, as a range of
/// its samples.
///
/// A `volume` IS the range of its samples, wrapped in the grid that names them.
/// Like @ref tf::points and @ref tf::polygons it takes only a policy and
/// inherits from @ref tf::form, so a policy can be tagged onto it without
/// moving a sample, and its dimensionality is deduced from that policy through
/// @ref tf::coordinate_dims_v. The samples are the only referenced memory; the
/// grid metadata (dims, spacing, origin) is tiny and held by value, so a view
/// is a value — copy it freely. @ref tf::volume_buffer is the owning carrier
/// and yields this view.
///
/// A volume is a sampled function: the samples are its codomain — what was
/// measured — and the grid is its domain — where the samples stand. The type
/// therefore answers two questions, `sample_type` and `coordinate_type`, and
/// they are the same by default: a `float` field stands on a `float` grid
/// unless the carrier states otherwise (@ref tf::volume_buffer) or the
/// factory deduces otherwise from the spacing and origin it is given — an
/// int16 CT on a millimetre grid is the stated case. The type a call decides
/// and emits in is the CALL's request (see @ref tf::make_isosurface),
/// resolved from the coordinate type when the caller states none; a sample
/// is read through one cast into it (@ref tf::volume_detail::field_value).
///
/// A 3D volume is the voxel grid @ref tf::make_isosurface reads; a 2D one is
/// the pixel grid @ref tf::make_isocontours reads, and is what a slice of a 3D
/// volume is (@ref tf::make_volume_slice). Nothing but the axis count differs.
///
/// The grid is axis-aligned in its own local frame: the mapping from an
/// integer sample index to a local-space point is the affine scale-plus-offset
///
///     local = origin + index * spacing
///
/// There is deliberately no rotation in this mapping — keeping the grid
/// axis-aligned collapses all index/physical math to a diagonal map. World
/// placement is the frame the volume states, like any form: tag one with
/// `vol | tf::tag(frame)` and the geometry-emitting entries
/// (@ref tf::make_isosurface, @ref tf::make_isocontours) emit through it,
/// while positional inputs stay local-space.
///
/// Samples are stored with the first axis varying fastest:
///
///     linear_index(x, y, z) = x + dims[0] * (y + dims[1] * z)
///
/// This matches the row-oriented traversal used by the flying-edges isosurface
/// extractor, which sweeps along x within each row.
///
/// The field is commonly interpreted as a signed distance field (SDF): the
/// surface lies at sample value 0, with negative inside and positive outside.
///
/// @tparam Policy The underlying grid policy over the samples.
template <typename Policy>
struct volume : form<coordinate_dims_v<Policy>, Policy> {
  using base = form<coordinate_dims_v<Policy>, Policy>;
  /// @brief What a sample is — the field's codomain scalar.
  using sample_type = typename Policy::sample_type;
  /// @brief How many axes the grid has.
  static constexpr std::size_t dimensions = coordinate_dims_v<Policy>;

  volume(const Policy &grid) : base{grid} {}
  volume(Policy &&grid) : base{std::move(grid)} {}

  /// @brief Total number of samples (the product of `dims()`).
  auto voxel_count() const -> std::size_t { return this->size(); }

  /// @brief Flat sample index, first axis fastest.
  auto linear_index(const std::array<int, dimensions> &at) const
      -> std::size_t {
    std::size_t flat = 0;
    for (std::size_t d = dimensions; d-- > 0;)
      flat = flat * static_cast<std::size_t>(this->dims()[d]) +
             static_cast<std::size_t>(at[d]);
    return flat;
  }

  /// @brief Flat sample index from one integer per axis.
  /// @overload
  template <typename... Ints>
  auto linear_index(Ints... c) const -> std::size_t {
    static_assert(sizeof...(Ints) == dimensions,
                  "a volume is indexed by one integer per axis");
    return linear_index(std::array<int, dimensions>{static_cast<int>(c)...});
  }

  /// @brief Sample at the given index.
  auto operator()(const std::array<int, dimensions> &at) const
      -> decltype(auto) {
    return (*this)[linear_index(at)];
  }

  /// @brief Sample at the given index.
  /// @overload
  template <typename... Ints>
  auto operator()(Ints... c) const -> decltype(auto) {
    return (*this)[linear_index(c...)];
  }

  /// @brief Local-space point at the given index: `origin + index * spacing`.
  ///
  /// @tparam Real The type the point is computed and returned in; the grid is
  ///         read through one cast into it.
  template <typename Real = tf::coordinate_type<Policy>>
  auto point_at(const std::array<int, dimensions> &at) const
      -> tf::point<Real, dimensions> {
    tf::point<Real, dimensions> p;
    for (std::size_t d = 0; d < dimensions; ++d)
      p[d] = static_cast<Real>(this->origin()[d]) +
             static_cast<Real>(at[d]) * static_cast<Real>(this->spacing()[d]);
    return p;
  }

  /// @brief Local-space point from one integer per axis.
  /// @overload
  template <typename Real = tf::coordinate_type<Policy>, typename... Ints>
  auto point_at(Ints... c) const -> tf::point<Real, dimensions> {
    static_assert(sizeof...(Ints) == dimensions,
                  "a volume is indexed by one integer per axis");
    return point_at<Real>(std::array<int, dimensions>{static_cast<int>(c)...});
  }
};

template <typename Policy>
auto unwrap(const volume<Policy> &vol) -> decltype(auto) {
  return static_cast<const Policy &>(vol);
}

template <typename Policy> auto unwrap(volume<Policy> &vol) -> decltype(auto) {
  return static_cast<Policy &>(vol);
}

template <typename Policy> auto unwrap(volume<Policy> &&vol) -> decltype(auto) {
  return static_cast<Policy &&>(vol);
}

template <typename Policy, typename T>
auto wrap_like(const volume<Policy> &, T &&t) {
  return volume<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(volume<Policy> &, T &&t) {
  return volume<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(volume<Policy> &&, T &&t) {
  return volume<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup volume
/// @brief View the product of `dims` samples as a scalar field on that grid.
///
/// The samples give `sample_type`; the spacing and origin give the grid's
/// coordinate type — pass float points over int16 samples and the view
/// carries both types with nothing spelled.
///
/// @param samples The flat sample storage (first axis fastest). Must outlive
///        the view.
/// @param dims Number of samples along each axis.
/// @param spacing Physical size of one grid step along each axis.
/// @param origin Local-space position of sample (0, ..., 0).
template <typename T, std::size_t Dims, typename P0, typename P1>
auto make_volume(T *samples, std::array<int, Dims> dims,
                 const tf::point_like<Dims, P0> &spacing,
                 const tf::point_like<Dims, P1> &origin) {
  using Coord = tf::coordinate_type<P0, P1>;
  auto r = tf::make_range(
      samples, samples + tf::volume_detail::grid_sample_count(dims));
  using policy_t = tf::volume_detail::grid<decltype(r), Coord, Dims>;
  return tf::volume<policy_t>{policy_t{r, dims,
                                       spacing.template as<Coord>(),
                                       origin.template as<Coord>()}};
}

/// @ingroup volume
/// @brief View the product of `dims` samples on a unit grid at the origin.
/// @overload
template <typename T, std::size_t Dims>
auto make_volume(T *samples, std::array<int, Dims> dims) {
  using Real = std::remove_const_t<T>;
  tf::point<Real, Dims> spacing, origin;
  for (std::size_t d = 0; d < Dims; ++d) {
    spacing[d] = Real{1};
    origin[d] = Real{0};
  }
  return tf::make_volume(samples, dims, spacing, origin);
}

} // namespace tf
