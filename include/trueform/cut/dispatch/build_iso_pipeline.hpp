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

#include "../../intersect/scalar_field_intersections.hpp"
#include "../classification/scalar_labels.hpp"
#include "../face_cuts.hpp"
#include "../partition/make_ids.hpp"

namespace tf::cut::dispatch {

template <typename Index, typename Policy, typename Scalars, typename Iterator,
          std::size_t N>
auto build_iso_pipeline(const tf::polygons<Policy> &polygons,
                        const Scalars &scalars,
                        const tf::range<Iterator, N> &cut_values) {
  tf::buffer<std::decay_t<decltype(cut_values[0])>> cut_vals;
  cut_vals.reserve(cut_values.size());
  std::copy(cut_values.begin(), cut_values.end(),
            std::back_inserter(cut_vals));
  std::sort(cut_vals.begin(), cut_vals.end());

  tf::scalar_field_intersections<Index, tf::coordinate_type<Policy>,
                                 tf::coordinate_dims_v<Policy>>
      sfi;
  sfi.build_many(polygons, scalars, cut_vals);

  tf::face_cuts<Index, tf::exact::int32> fc;
  fc.build(polygons, sfi);

  auto labels = tf::cut::make_scalar_labels<Index>(
      polygons, sfi, fc, scalars, tf::make_range(cut_vals));
  auto pids = tf::cut::make_partition_ids<Index>(labels);

  return std::make_tuple(std::move(sfi), std::move(fc), std::move(pids));
}

} // namespace tf::cut::dispatch
