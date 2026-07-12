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

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/zip.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../intersect/types/simple_intersections.hpp"
#include "../face_cuts.hpp"
#include "../partition/partition_labels.hpp"
#include "tbb/parallel_invoke.h"

namespace tf::cut {

/// Classify cut face loops by scalar band.
/// A loop lies strictly inside one band, so its first original vertex not
/// sitting on a cut value decides. A loop without one is bounded by cut
/// chords and on-cut vertices alone; its band is the highest bounding cut
/// index.
template <typename LabelType, typename Index, typename Int,
          typename GetCategory, typename GetOnCut, typename GetCreatedCut>
auto make_cut_scalar_labels(const tf::face_cuts<Index, Int> &fc,
                            const GetCategory &get_category,
                            const GetOnCut &get_on_cut,
                            const GetCreatedCut &get_created_cut) {
  auto loops = fc.loops();
  tf::buffer<LabelType> out;
  out.allocate(loops.size());
  tf::parallel_for(
      tf::zip(loops, out),
      [&](auto begin, auto end) {
        for (auto &&[loop, label] : tf::make_range(begin, end)) {
          bool decided = false;
          for (const auto &v : loop) {
            if (v.source == tf::intersect::graph::vertex_source::original &&
                !get_on_cut(v.id)) {
              label = get_category(v.id);
              decided = true;
              break;
            }
          }
          if (decided)
            continue;
          LabelType band = 0;
          for (const auto &v : loop)
            band = std::max(
                band,
                v.source == tf::intersect::graph::vertex_source::created
                    ? LabelType(get_created_cut(v.id))
                    : LabelType(get_category(v.id)));
          label = band;
        }
      },
      tf::checked);
  return out;
}

/// Classify uncut surface faces by scalar band.
/// Cut faces get label -1. An uncut face takes its maximum vertex category:
/// a vertex sitting exactly on a cut value reports the band below, while
/// the face interior lies in the band above.
template <typename LabelType, typename Index, typename Policy, typename RealT,
          std::size_t Dims, typename GetCategory>
auto make_surface_scalar_labels(
    const tf::polygons<Policy> &polygons,
    const tf::intersect::simple_intersections<Index, RealT, Dims> &si,
    const GetCategory &get_category) {
  tf::buffer<LabelType> out;
  out.allocate(polygons.size());
  tf::parallel_fill(out, LabelType(-2));
  tf::parallel_for_each(
      si.intersections(), [&](const auto &r) { out[r.front().object] = -1; },
      tf::checked);
  tf::parallel_for_each(
      tf::zip(out, polygons.faces()),
      [&](auto pair) {
        auto &&[label, face] = pair;
        if (label == -1)
          return;
        label = get_category(face[0]);
        for (std::size_t i = 1; i < std::size_t(face.size()); ++i)
          label = std::max(label, LabelType(get_category(face[i])));
      },
      tf::checked);
  return out;
}

/// Build scalar labels for both cut and uncut faces.
/// Returns partition_labels with n_components = n_cut_values + 1.
template <typename LabelType, typename Index, typename Policy, typename RealT,
          std::size_t Dims, typename Int, typename Scalars, typename Iterator,
          std::size_t N>
auto make_scalar_labels(
    const tf::polygons<Policy> &polygons,
    const tf::intersect::simple_intersections<Index, RealT, Dims> &si,
    const tf::face_cuts<Index, Int> &fc, const Scalars &scalars,
    tf::range<Iterator, N> cut_values) {
  tf::buffer<LabelType> categories;
  categories.allocate(scalars.size());
  tf::parallel_for_each(tf::zip(scalars, categories), [&](auto pair) {
    auto &&[scalar, category] = pair;
    category = std::lower_bound(cut_values.begin(), cut_values.end(), scalar) -
               cut_values.begin();
  });
  auto get_category = [&](auto x) { return categories[x]; };
  auto get_on_cut = [&](auto x) {
    auto category = categories[x];
    return std::size_t(category) < cut_values.size() &&
           scalars[x] == cut_values[category];
  };
  auto point_cuts = si.intersection_point_cuts();
  auto get_created_cut = [&](auto x) { return point_cuts[x]; };

  tf::cut::partition_labels<LabelType> lbls;
  lbls.n_components = cut_values.size() + 1;

  tbb::parallel_invoke(
      [&] {
        lbls.polygon_labels =
            make_surface_scalar_labels<LabelType>(polygons, si, get_category);
      },
      [&] {
        lbls.cut_labels = make_cut_scalar_labels<LabelType, Index>(
            fc, get_category, get_on_cut, get_created_cut);
      });
  return lbls;
}

} // namespace tf::cut
