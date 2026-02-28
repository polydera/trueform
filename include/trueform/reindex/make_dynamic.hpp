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
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/algorithm/parallel_transform.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/polygons.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/static_size.hpp"
#include "../core/views/sequence_range.hpp"
#include "../core/views/zip.hpp"

namespace tf {

/// @ingroup reindex
/// @brief Convert a polygons buffer to dynamic-size (rvalue, moves data).
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto make_dynamic(tf::polygons_buffer<Index, Real, Dims, Ngon> &&polygons)
    -> tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size> {
  tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size> out;
  if (polygons.size() == 0)
    return out;
  out.faces_buffer().data_buffer() =
      std::move(polygons.faces_buffer().data_buffer());
  out.points_buffer().data_buffer() =
      std::move(polygons.points_buffer().data_buffer());
  if constexpr (Ngon == tf::dynamic_size)
    out.faces_buffer().offsets_buffer() =
        std::move(polygons.faces_buffer().offsets_buffer());
  else {
    out.faces_buffer().offsets_buffer().allocate(polygons.faces().size() + 1);
    tf::parallel_transform(
        tf::make_sequence_range(
            static_cast<Index>(out.faces_buffer().offsets_buffer().size())),
        out.faces_buffer().offsets_buffer(),
        [](auto i) -> Index { return static_cast<Index>(i * Ngon); });
  }
  return out;
}

/// @ingroup reindex
/// @brief Convert a polygons buffer to dynamic-size (const, copies data).
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto make_dynamic(const tf::polygons_buffer<Index, Real, Dims, Ngon> &polygons)
    -> tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size> {
  tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size> out;
  if (polygons.size() == 0)
    return out;
  out.faces_buffer().data_buffer().allocate(
      polygons.faces_buffer().data_buffer().size());
  tf::parallel_copy(polygons.faces_buffer().data_buffer(),
                    out.faces_buffer().data_buffer());
  out.points_buffer().data_buffer().allocate(
      polygons.points_buffer().data_buffer().size());
  tf::parallel_copy(polygons.points_buffer().data_buffer(),
                    out.points_buffer().data_buffer());
  if constexpr (Ngon == tf::dynamic_size) {
    out.faces_buffer().offsets_buffer().allocate(
        polygons.faces_buffer().offsets_buffer().size());
    tf::parallel_copy(polygons.faces_buffer().offsets_buffer(),
                      out.faces_buffer().offsets_buffer());
  } else {
    out.faces_buffer().offsets_buffer().allocate(polygons.faces().size() + 1);
    tf::parallel_transform(
        tf::make_sequence_range(
            static_cast<Index>(out.faces_buffer().offsets_buffer().size())),
        out.faces_buffer().offsets_buffer(),
        [](auto i) -> Index { return static_cast<Index>(i * Ngon); });
  }
  return out;
}

/// @ingroup reindex
/// @brief Convert a polygons view to a dynamic-size buffer.
template <typename Policy>
auto make_dynamic(const tf::polygons<Policy> &polygons) {
  using index_t = std::decay_t<decltype(polygons.faces()[0][0])>;
  using real_t = tf::coordinate_type<Policy>;
  constexpr auto dims = tf::coordinate_dims_v<Policy>;
  constexpr auto ngon =
      tf::static_size_v<std::decay_t<decltype(polygons.faces()[0])>>;

  tf::polygons_buffer<index_t, real_t, dims, tf::dynamic_size> out;
  if (polygons.size() == 0)
    return out;

  const auto n_faces = polygons.faces().size();

  // Build offsets
  out.faces_buffer().offsets_buffer().allocate(n_faces + 1);
  if constexpr (ngon == tf::dynamic_size) {
    // Variable sizes — sequential cumulative sum
    auto &offsets = out.faces_buffer().offsets_buffer();
    offsets[0] = 0;
    index_t acc = 0;
    for (std::size_t i = 0; i < n_faces; ++i) {
      acc += static_cast<index_t>(polygons.faces()[i].size());
      offsets[i + 1] = acc;
    }
  } else {
    tf::parallel_transform(
        tf::make_sequence_range(
            static_cast<index_t>(out.faces_buffer().offsets_buffer().size())),
        out.faces_buffer().offsets_buffer(),
        [](auto i) -> index_t { return static_cast<index_t>(i * ngon); });
  }

  // Allocate and copy face indices
  out.faces_buffer().data_buffer().allocate(
      out.faces_buffer().offsets_buffer().back());
  tf::parallel_for_each(
      tf::zip(polygons.faces(), out.faces_buffer()),
      [](auto pair) {
        auto &&[in_face, out_face] = pair;
        for (auto &&[v_in, v_out] : tf::zip(in_face, out_face))
          v_out = v_in;
      },
      tf::checked);

  // Copy points
  out.points_buffer().allocate(polygons.points().size());
  tf::parallel_copy(polygons.points(), out.points_buffer());

  return out;
}

} // namespace tf
