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
#include "../../core/algorithm/block_reduce.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"
#include "../../core/cross.hpp"
#include "../../core/dot.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../exact/determinant.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/vertex.hpp"
#include <type_traits>

namespace tf::topology::domains {

/// @ingroup topology_components
/// @brief Per-MEL-fragment Gauss volume contributions (no `1/6` factor).
///
/// Sums the divergence-theorem integrand
/// `dot(p0, cross(p1, p2))` over each face's fan-triangulation, keyed
/// by the face's MEL fragment id. For a closed fragment the result is
/// `6 V` where `V` is the signed enclosed volume; positive when the
/// stored winding faces outward. For open fragments the sum is not
/// translation-invariant and therefore meaningless --- callers must
/// mask them out before consuming.
///
/// Two implementations selected by `if constexpr` on the polygons'
/// coordinate type:
///
/// - **Integer coords** (`tf::exact::int32` / `int64`): per-triangle
///   contribution computed by @ref tf::exact::determinant_value, summed
///   into a per-fragment accumulator of @ref tf::exact::meta::T2 (e.g.
///   `int128` for `int32` input). Translation-invariance of the closed
///   sum lets us evaluate at the original coordinates without a
///   reference subtraction; intermediate products do not overflow `T1`,
///   the accumulator does not overflow `T2` for any practical mesh.
///
/// - **Floating coords**: per-triangle contribution computed via
///   `tf::dot` / `tf::cross` after subtracting the first mesh point as
///   reference (translation-invariant, improves cancellation), summed
///   into a per-fragment `double` accumulator.
///
/// The aggregation pattern is `tf::blocked_reduce`
/// with a per-task per-fragment buffer; no atomics.
///
/// @param polygons The polygons range.
/// @param fragment_labels Per-face MEL fragment labels.
/// @return `tf::buffer<T>` of size `fragment_labels.n_components`,
///         where `T` is `tf::exact::meta<CoordT>::T2` for integer
///         coords or `double` for floating coords.
template <typename Polygons, typename FragLabels>
auto compute_volume_contributions_per_fragment(
    const Polygons &polygons, const FragLabels &fragment_labels) {
  using CoordT = tf::coordinate_type<Polygons>;

  if constexpr (std::is_integral_v<CoordT>) {
    using AccumT = typename tf::exact::meta<CoordT>::T2;
    tf::buffer<AccumT> result;
    result.allocate(fragment_labels.n_components);
    tf::parallel_fill(result, AccumT(0));

    if (polygons.size() == 0)
      return result;

    tf::buffer<AccumT> prototype;
    prototype.allocate(fragment_labels.n_components);
    tf::parallel_fill(prototype, AccumT(0));

    tf::blocked_reduce(
        tf::enumerate(polygons), result, prototype,
        [&](auto &&block, tf::buffer<AccumT> &local) {
          for (const auto &pair : block) {
            const auto &[f, poly] = pair;
            tf::exact::pt3<CoordT> p0{CoordT(poly[0][0]), CoordT(poly[0][1]),
                                      CoordT(poly[0][2])};
            auto size = poly.size();
            AccumT sum(0);
            for (decltype(size) i = 1; i + 1 < size; ++i) {
              tf::exact::pt3<CoordT> p1{CoordT(poly[i][0]), CoordT(poly[i][1]),
                                        CoordT(poly[i][2])};
              tf::exact::pt3<CoordT> p2{CoordT(poly[i + 1][0]),
                                        CoordT(poly[i + 1][1]),
                                        CoordT(poly[i + 1][2])};
              sum += tf::exact::determinant_value(p0, p1, p2);
            }
            local[fragment_labels.labels[f]] += sum;
          }
        },
        [](const tf::buffer<AccumT> &local, tf::buffer<AccumT> &out) {
          for (decltype(out.size()) i = 0; i < out.size(); ++i)
            out[i] += local[i];
        });

    return result;
  } else {
    tf::buffer<double> result;
    result.allocate(fragment_labels.n_components);
    tf::parallel_fill(result, 0.0);

    if (polygons.size() == 0)
      return result;

    auto frame = tf::frame_of(polygons);
    auto ref = tf::transformed(polygons.points()[0], frame);

    tf::buffer<double> prototype;
    prototype.allocate(fragment_labels.n_components);
    tf::parallel_fill(prototype, 0.0);

    tf::blocked_reduce(
        tf::enumerate(polygons), result, prototype,
        [&](auto &&block, tf::buffer<double> &local) {
          for (const auto &pair : block) {
            const auto &[f, poly] = pair;
            auto v0 = tf::transformed(poly[0], frame) - ref;
            auto size = poly.size();
            double sum = 0;
            for (decltype(size) i = 1; i + 1 < size; ++i) {
              auto v1 = tf::transformed(poly[i], frame) - ref;
              auto v2 = tf::transformed(poly[i + 1], frame) - ref;
              sum += double(tf::dot(v0, tf::cross(v1, v2)));
            }
            local[fragment_labels.labels[f]] += sum;
          }
        },
        [](const tf::buffer<double> &local, tf::buffer<double> &out) {
          for (decltype(out.size()) i = 0; i < out.size(); ++i)
            out[i] += local[i];
        });

    return result;
  }
}

} // namespace tf::topology::domains
