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
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_transform.hpp"
#include "../core/buffer.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/none.hpp"
#include "../core/point.hpp"
#include "../core/resolved_output_real.hpp"
#include "./impl/field_value.hpp"
#include "./impl/grid_sample_count.hpp"
#include "./volume.hpp"
#include <array>
#include <cstddef>
#include <type_traits>

namespace tf {

/// @ingroup volume
/// @brief An owning buffer of a dense, axis-aligned scalar field on a grid.
///
/// Stores the samples in a flat @ref tf::buffer plus the grid metadata
/// (dims, spacing, origin) — and nothing else: a pose belongs to the view's
/// frame tag and stays with the caller. Use `volume()` to obtain the
/// non-owning @ref tf::volume view every volume algorithm consumes. Sample
/// addressing and index/physical math are stated once, on the view, and
/// forwarded here.
///
/// @tparam T The sample scalar type — what a sample is.
/// @tparam Coord The grid's coordinate type — where samples stand
///         (default: `T`, the same-by-default law of @ref tf::volume).
/// @tparam Dims The number of grid axes.
template <typename T, typename Coord = T, std::size_t Dims = 3>
class volume_buffer {
public:
  using sample_type = T;
  using coordinate_type = Coord;
  using grid_dims = std::integral_constant<std::size_t, Dims>;

  volume_buffer() {
    for (std::size_t d = 0; d < Dims; ++d) {
      _spacing[d] = Coord{1};
      _origin[d] = Coord{0};
    }
  }

  /// @brief Construct a volume buffer with the given grid resolution.
  ///
  /// Allocates the product of `dims` samples (uninitialized). Spacing defaults
  /// to unit steps and origin to the local frame origin; set them via
  /// @ref set_spacing / @ref set_origin.
  explicit volume_buffer(std::array<int, Dims> dims) : volume_buffer() {
    allocate(dims);
  }

  /// @brief Construct a fully-specified volume buffer.
  ///
  /// @param dims Number of samples along each axis.
  /// @param spacing Physical size of one grid step along each axis.
  /// @param origin Local-space position of sample (0, ..., 0).
  volume_buffer(std::array<int, Dims> dims, tf::point<Coord, Dims> spacing,
                tf::point<Coord, Dims> origin)
      : _spacing(spacing), _origin(origin) {
    allocate(dims);
  }

  /// @brief (Re)allocate the sample buffer for `dims` resolution.
  ///
  /// Leaves spacing and origin unchanged. Samples are left uninitialized.
  auto allocate(std::array<int, Dims> dims) -> void {
    _dims = dims;
    _samples.allocate(tf::volume_detail::grid_sample_count(_dims));
  }

  /// @brief Number of samples along each axis.
  auto dims() const -> const std::array<int, Dims> & { return _dims; }

  /// @brief Physical size of one grid step along each axis.
  auto spacing() const -> const tf::point<Coord, Dims> & { return _spacing; }
  auto set_spacing(tf::point<Coord, Dims> spacing) -> void {
    _spacing = spacing;
  }

  /// @brief Local-space position of sample (0, ..., 0).
  auto origin() const -> const tf::point<Coord, Dims> & { return _origin; }
  auto set_origin(tf::point<Coord, Dims> origin) -> void { _origin = origin; }

  /// @brief The flat sample storage (first axis fastest).
  auto samples_buffer() -> tf::buffer<T> & { return _samples; }
  auto samples_buffer() const -> const tf::buffer<T> & { return _samples; }

  /// @brief Total number of samples (the product of `dims`).
  auto voxel_count() const -> std::size_t { return _samples.size(); }

  /// @brief Flat sample index, first axis fastest.
  template <typename... Ints>
  auto linear_index(Ints... c) const -> std::size_t {
    return volume().linear_index(c...);
  }

  /// @brief Sample at the given index.
  template <typename... Ints> auto operator()(Ints... c) const -> const T & {
    return volume()(c...);
  }
  template <typename... Ints> auto operator()(Ints... c) -> T & {
    return volume()(c...);
  }

  /// @brief Local-space point at the given index: `origin + index * spacing`.
  template <typename Real = Coord, typename... Ints>
  auto point_at(Ints... c) const -> tf::point<Real, Dims> {
    return volume().template point_at<Real>(c...);
  }

  /// @brief The non-owning view over this buffer's samples and metadata.
  auto volume() {
    return tf::make_volume(_samples.data(), _dims, _spacing, _origin);
  }
  auto volume() const {
    return tf::make_volume(_samples.data(), _dims, _spacing, _origin);
  }

private:
  std::array<int, Dims> _dims{};
  tf::point<Coord, Dims> _spacing;
  tf::point<Coord, Dims> _origin;
  tf::buffer<T> _samples;
};

/// @ingroup volume
/// @brief Create a volume buffer holding a copy of a volume view's field.
///
/// @tparam OutputCoordinateType The type the copy decides in: stated, it
///         becomes both the sample and coordinate type; unstated, the copy
///         preserves the view's own two types.
template <typename OutputCoordinateType = tf::none_t, typename Policy>
auto make_volume_buffer(const tf::volume<Policy> &vol) {
  constexpr std::size_t Dims = tf::coordinate_dims_v<Policy>;
  if constexpr (std::is_same_v<OutputCoordinateType, tf::none_t>) {
    using Sample = typename tf::volume<Policy>::sample_type;
    using Coord = tf::coordinate_type<Policy>;
    tf::volume_buffer<Sample, Coord, Dims> out(vol.dims(), vol.spacing(),
                                               vol.origin());
    tf::parallel_copy(vol, out.samples_buffer());
    return out;
  } else {
    using RealOut = tf::resolved_output_real_t<OutputCoordinateType,
                                               tf::coordinate_type<Policy>>;
    tf::volume_buffer<RealOut, RealOut, Dims> out(
        vol.dims(), vol.spacing().template as<RealOut>(),
        vol.origin().template as<RealOut>());
    tf::parallel_transform(vol, out.samples_buffer(), [](const auto &sample) {
      return volume_detail::field_value<RealOut>(sample);
    });
    return out;
  }
}

} // namespace tf
