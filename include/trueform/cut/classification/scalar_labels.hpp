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
#include "../../core/small_vector.hpp"
#include "../../core/views/zip.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../intersect/types/simple_intersections.hpp"
#include "../face_cuts.hpp"
#include "../partition/partition_labels.hpp"
#include "tbb/parallel_invoke.h"

namespace tf::cut {

/// Classify cut face loops by scalar band.
/// For each loop, counts original vertices per category (majority vote).
template <typename LabelType, typename Index, typename Int,
          typename GetCategory>
auto make_cut_scalar_labels(const tf::face_cuts<Index, Int> &fc,
                            const GetCategory &get_category,
                            std::size_t n_categories) {
  auto loops = fc.loops();
  tf::buffer<LabelType> out;
  out.allocate(loops.size());
  tf::parallel_for(
      tf::zip(loops, out),
      [&](auto begin, auto end) {
        tf::small_vector<LabelType, 20> counts;
        for (auto &&[loop, label] : tf::make_range(begin, end)) {
          counts.clear();
          counts.resize(n_categories);
          for (const auto &v : loop) {
            if (v.source == tf::intersect::graph::vertex_source::original)
              counts[get_category(v.id)]++;
          }
          label =
              std::max_element(counts.begin(), counts.end()) - counts.begin();
        }
      },
      tf::checked);
  return out;
}

/// Classify uncut surface faces by scalar band.
/// Cut faces get label -1, uncut faces get category of first vertex.
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
        if (label != -1)
          label = get_category(face[0]);
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

  tf::cut::partition_labels<LabelType> lbls;
  lbls.n_components = cut_values.size() + 1;

  tbb::parallel_invoke(
      [&] {
        lbls.polygon_labels =
            make_surface_scalar_labels<LabelType>(polygons, si, get_category);
      },
      [&] {
        lbls.cut_labels = make_cut_scalar_labels<LabelType, Index>(
            fc, get_category, cut_values.size() + 1);
      });
  return lbls;
}

} // namespace tf::cut
