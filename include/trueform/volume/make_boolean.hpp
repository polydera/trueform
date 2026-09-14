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
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/frame.hpp"
#include "../core/none.hpp"
#include "../core/point.hpp"
#include "../core/policy/frame.hpp"
#include "../core/resolved_output_real.hpp"
#include "../core/transformation.hpp"
#include "../core/transformed.hpp"
#include "../core/views/sequence_range.hpp"
#include "./boolean_op.hpp" // IWYU pragma: export
#include "./impl/combine_sdf.hpp"
#include "./impl/field_value.hpp"
#include "./impl/grid_sample_count.hpp"
#include "./impl/grids_match.hpp"
#include "./make_resampled_volume.hpp"
#include "./volume.hpp"
#include "./volume_buffer.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup volume
/// @brief Boolean CSG of two signed distance fields, returned as a new field.
///
/// The volume overload of the library's boolean: it combines two SDFs
/// sample-wise under `op` (union / intersection / difference) using the
/// negative-inside convention. The zero level set of the result is exactly the
/// boolean of the two input solids, so extracting its isosurface (via
/// @ref tf::make_isosurface) yields the CSG surface. The algebra is the
/// field's, not the grid's, so it holds at any number of axes.
///
/// The combine happens on one shared grid. Matched grids AND matched poses
/// combine directly — the result carries the shared pose. Anything else —
/// different grids, different poses — is resampled onto the union of the
/// WORLD domains first, through @ref tf::make_resampled_volume, at the finer
/// of the two LOCAL spacings (a scaling pose does not enter that election)
/// and spanning the union to within one truncated step; the result is
/// world-axis-aligned. Pose equality is exact matrix equality in the type the
/// call decides in, so poses differing below that type's resolution are one
/// pose. The output sample count is capped so far-apart fine grids cannot
/// exhaust memory.
///
/// Out of an operand's domain is outside its solid: on the shared grid every
/// operand, posed or not, resamples through its pose and reads the
/// far-outside sentinel past its own box, so no operand's field is continued
/// past the domain it was sampled on.
///
/// An empty operand is the empty set and the algebra answers for it:
/// `A ∪ ∅ = A`, `A ∩ ∅ = ∅`, `A \ ∅ = A`, `∅ \ B = ∅`.
///
/// The call combines and emits in ONE type: `OutputCoordinateType`, or A's
/// coordinate type when the caller states none. The combine is refused at
/// compile time for an unsigned sample type — an SDF has a negative inside —
/// and for any non-floating deciding type, B's coordinates included.
///
/// @tparam OutputCoordinateType The sample type of the combined field
///         (default: A's coordinate type).
/// @param a The first field (A).
/// @param b The second field (B).
/// @param op The CSG operation.
/// @return Unposed operands: a `volume_buffer` holding the combined field,
///         as before. Any posed operand: the shape changes loudly at the
///         call site — the field and the pose it stands in, the shared pose
///         on the matched fast path and identity for the world-axis-aligned
///         general result.
template <typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_boolean(const tf::volume<Policy0> &a, const tf::volume<Policy1> &b,
                  volume_boolean_op op) {
  constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;
  static_assert(tf::coordinate_dims_v<Policy1> == Dims,
                "both fields carry the same number of axes");
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType,
                                             tf::coordinate_type<Policy0>>;
  static_assert(
      !std::is_unsigned_v<typename tf::volume<Policy0>::sample_type> &&
          !std::is_unsigned_v<typename tf::volume<Policy1>::sample_type>,
      "an SDF has a negative inside; an unsigned field is not an SDF");
  static_assert(std::is_floating_point_v<RealOut>,
                "the combine decides in a floating type: state "
                "OutputCoordinateType, or give the field a floating "
                "coordinate type");
  static_assert(!std::is_integral_v<tf::coordinate_type<Policy1>> ||
                    !std::is_floating_point_v<RealOut>,
                "the combine decides across both grids: B's coordinate type "
                "must be floating too");
  const auto combined = [op](const auto &va, const auto &vb,
                             auto &out_buffer) {
    auto *out = out_buffer.samples_buffer().data();
    tf::parallel_for_each(
        tf::make_sequence_range(std::size_t{0}, va.voxel_count()),
        [&](std::size_t i) {
          out[i] = volume_detail::combine_sdf(
              op, volume_detail::field_value<RealOut>(va[i]),
              volume_detail::field_value<RealOut>(vb[i]));
        });
  };

  // One producer of the union-grid election: dims from the span, the cap
  // coarsening isotropically so far-apart fine grids cannot exhaust memory.
  const auto elect_dims =
      [](tf::point<RealOut, tf::coordinate_dims_v<Policy0>> &spacing,
         const tf::point<RealOut, tf::coordinate_dims_v<Policy0>> &lo,
         const tf::point<RealOut, tf::coordinate_dims_v<Policy0>> &hi) {
    std::array<int, tf::coordinate_dims_v<Policy0>> dims;
    const auto span = [&](std::size_t d) {
      const RealOut extent = std::max(hi[d] - lo[d], RealOut{0});
      return std::max(2, static_cast<int>(std::floor(extent / spacing[d])) + 1);
    };
    for (std::size_t d = 0; d < tf::coordinate_dims_v<Policy0>; ++d)
      dims[d] = span(d);
    const std::size_t kMaxSamples = std::size_t{1} << 26; // ~67M
    while (tf::volume_detail::grid_sample_count(dims) > kMaxSamples)
      for (std::size_t d = 0; d < tf::coordinate_dims_v<Policy0>; ++d) {
        spacing[d] *= static_cast<RealOut>(2);
        dims[d] = span(d);
      }
    return dims;
  };

  // The answer's shape: a posed operand makes the pose part of the answer,
  // and the tuple says so at the call site.
  const auto shaped = [](auto &&out, const auto &pose) {
    if constexpr (tf::has_frame_policy<Policy0> ||
                  tf::has_frame_policy<Policy1>)
      return std::make_tuple(std::move(out), pose);
    else
      return std::move(out);
  };

  // The pose an operand stands in, as a concrete matrix; untagged means
  // identity, so one path serves every combination — pushing a local box
  // through an identity pose reproduces the box.
  const auto pose_a = [&] {
    auto out = tf::make_identity_transformation<RealOut,
                                                tf::coordinate_dims_v<Policy0>>();
    if constexpr (tf::has_frame_policy<Policy0>) {
      const auto &t = a.frame().transformation();
      for (std::size_t i = 0; i < tf::coordinate_dims_v<Policy0>; ++i)
        for (std::size_t j = 0; j < tf::coordinate_dims_v<Policy0> + 1; ++j)
          out(i, j) = static_cast<RealOut>(t(i, j));
    }
    return out;
  }();
  const auto pose_b = [&] {
    auto out = tf::make_identity_transformation<RealOut,
                                                tf::coordinate_dims_v<Policy1>>();
    if constexpr (tf::has_frame_policy<Policy1>) {
      const auto &t = b.frame().transformation();
      for (std::size_t i = 0; i < tf::coordinate_dims_v<Policy1>; ++i)
        for (std::size_t j = 0; j < tf::coordinate_dims_v<Policy1> + 1; ++j)
          out(i, j) = static_cast<RealOut>(t(i, j));
    }
    return out;
  }();

  // A degenerate operand is the empty set, and the algebra names which
  // operand the answer copies: the empty one wherever the answer is empty.
  if (a.voxel_count() == 0) {
    if (op == volume_boolean_op::union_)
      return shaped(tf::make_volume_buffer<RealOut>(b), pose_b);
    return shaped(tf::make_volume_buffer<RealOut>(a), pose_a);
  }
  if (b.voxel_count() == 0) {
    if (op == volume_boolean_op::intersection)
      return shaped(tf::make_volume_buffer<RealOut>(b), pose_b);
    return shaped(tf::make_volume_buffer<RealOut>(a), pose_a);
  }

  // Fast path: one grid, one pose -> the combine is elementwise on the
  // samples, the grid is never addressed, and the shared pose carries.
  bool poses_match = true;
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < Dims + 1; ++j)
      poses_match = poses_match && pose_a(i, j) == pose_b(i, j);
  if (poses_match && volume_detail::grids_match<RealOut>(a, b)) {
    tf::volume_buffer<RealOut, RealOut, Dims> out_buffer(
        a.dims(), a.spacing().template as<RealOut>(),
        a.origin().template as<RealOut>());
    combined(a, b, out_buffer);
    return shaped(std::move(out_buffer), pose_a);
  }

  // General path: the union of the world domains — every corner of each
  // local box through its pose — at the finer local spacing; both operands
  // resample through the one regrid, and the result is world-axis-aligned.
  tf::point<RealOut, Dims> lo, hi, spacing;
  for (std::size_t d = 0; d < Dims; ++d) {
    lo[d] = std::numeric_limits<RealOut>::max();
    hi[d] = std::numeric_limits<RealOut>::lowest();
    spacing[d] = std::min(static_cast<RealOut>(a.spacing()[d]),
                          static_cast<RealOut>(b.spacing()[d]));
  }
  const auto grow = [&](const auto &v, const auto &pose) {
    for (unsigned corner = 0; corner < (1u << tf::coordinate_dims_v<Policy0>);
         ++corner) {
      tf::point<RealOut, tf::coordinate_dims_v<Policy0>> p;
      for (std::size_t d = 0; d < tf::coordinate_dims_v<Policy0>; ++d) {
        const RealOut extent = static_cast<RealOut>(v.dims()[d] - 1) *
                               static_cast<RealOut>(v.spacing()[d]);
        p[d] = static_cast<RealOut>(v.origin()[d]) +
               (((corner >> d) & 1u) != 0 ? extent : RealOut{0});
      }
      const auto w = tf::transformed(p, pose);
      for (std::size_t d = 0; d < tf::coordinate_dims_v<Policy0>; ++d) {
        lo[d] = std::min(lo[d], w[d]);
        hi[d] = std::max(hi[d], w[d]);
      }
    }
  };
  grow(a, pose_a);
  grow(b, pose_b);
  const auto dims = elect_dims(spacing, lo, hi);

  // Out of an operand's domain is outside its solid, so both operands read
  // the far-outside sentinel there: each resamples through a pose, an
  // untagged one through the identity its own pose states.
  const auto resampled = [&](const auto &v, const auto &pose) {
    if constexpr (tf::has_frame_policy<decltype(v)>)
      return tf::make_resampled_volume<RealOut>(v, dims, spacing, lo);
    else
      return tf::make_resampled_volume<RealOut>(
          v | tf::tag(tf::make_frame(pose)), dims, spacing, lo);
  };
  const auto ra = resampled(a, pose_a);
  const auto rb = resampled(b, pose_b);
  tf::volume_buffer<RealOut, RealOut, Dims> out_buffer(dims, spacing, lo);
  combined(ra.volume(), rb.volume(), out_buffer);
  return shaped(std::move(out_buffer),
                tf::make_identity_transformation<RealOut, Dims>());
}

} // namespace tf
