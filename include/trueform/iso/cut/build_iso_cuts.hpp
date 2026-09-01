/*
 * Copyright (c) 2026 XLAB
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

#include "../../arrangement/partition/make_ids.hpp"
#include "../../arrangement/partition/partition_labels.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/coordinate_dims.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/none.hpp"
#include "../../core/polygons.hpp"
#include "../../core/range.hpp"
#include "../../core/views/zip.hpp"
#include "../../exact/resolve_int_type.hpp"
#include "../scalar_field_intersections.hpp"
#include "./cut_iso_faces.hpp"
#include "./iso_cut_regions.hpp"
#include "./make_surface_scalar_labels.hpp"
#include "tbb/parallel_invoke.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf::iso {

/// The scalar-field cut of one mesh: the field's crossings, the pieces they
/// cut every crossed face into, and the per-band id lists both entry points
/// read. A vertex category is `lower_bound(cut_values, scalar)`, so the same
/// classification serves the uncut surface and the cut pieces.
template <typename Index, typename Policy, typename Scalars, typename Iterator,
          std::size_t N>
auto build_iso_cuts(const tf::polygons<Policy> &polygons,
                    const Scalars &scalars,
                    const tf::range<Iterator, N> &cut_values) {
  using ResolvedInt =
      tf::exact::resolve_int_type<tf::none_t, tf::coordinate_type<Policy>>;

  tf::buffer<std::decay_t<decltype(cut_values[0])>> cut_vals;
  cut_vals.reserve(cut_values.size());
  std::copy(cut_values.begin(), cut_values.end(),
            std::back_inserter(cut_vals));
  std::sort(cut_vals.begin(), cut_vals.end());
  cut_vals.erase_till_end(std::unique(cut_vals.begin(), cut_vals.end()));

  tf::scalar_field_intersections<Index, tf::coordinate_type<Policy>,
                                 tf::coordinate_dims_v<Policy>>
      sfi;
  sfi.build_many(polygons, scalars, cut_vals);

  tf::buffer<Index> categories;
  categories.allocate(scalars.size());
  tf::parallel_for_each(tf::zip(scalars, categories), [&](auto pair) {
    auto &&[scalar, category] = pair;
    category = std::lower_bound(cut_vals.begin(), cut_vals.end(), scalar) -
               cut_vals.begin();
  });
  auto get_category = [&](auto x) { return categories[x]; };
  auto get_on_cut = [&](auto x) {
    auto category = categories[x];
    return std::size_t(category) < cut_vals.size() &&
           scalars[x] == cut_vals[category];
  };
  auto point_cuts = sfi.intersection_point_cuts();
  auto get_created_cut = [&](auto x) { return point_cuts[x]; };

  tf::arrangement::partition_labels<Index> labels;
  labels.n_components = static_cast<Index>(cut_vals.size() + 1);
  iso_cut_regions<Index, tf::coordinate_type<Policy>,
                  tf::coordinate_dims_v<Policy>>
      regions;

  tbb::parallel_invoke(
      [&] {
        labels.polygon_labels =
            make_surface_scalar_labels<Index>(polygons, sfi, get_category);
      },
      [&] {
        regions = cut_iso_faces<ResolvedInt>(polygons, sfi, get_category,
                                             get_on_cut, get_created_cut,
                                             labels.cut_labels);
      });
  auto pids = tf::arrangement::make_partition_ids<Index>(labels);

  return std::make_tuple(std::move(sfi), std::move(regions), std::move(pids));
}

} // namespace tf::iso
