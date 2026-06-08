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

#include "../preserve_regions.hpp"

namespace tf::remesh {

/// @brief The value type used for the per-face region labels a feature_handler
/// carries, deduced from the regions argument of a remesh call.
///
/// - tf::none_t (no regions): the mesh index type (the labels buffer stays
///   empty, so the type is unused).
/// - tf::preserve_regions_t<Range>: the input range's element type, so labels
///   round-trip in the exact type the caller passed -- no cast, no truncation
///   to the mesh index width.
///
/// This is a partial specialization rather than std::conditional_t on purpose:
/// conditional_t requires BOTH type arguments to be well-formed, so probing the
/// range's element type would also be instantiated for none_t (which has no
/// range) and fail to compile. With a specialization, the range probe is only
/// ever formed for the preserve_regions_t case.
template <typename Regions, typename Index>
struct region_label {
  using type = Index;
};

template <typename Range, typename Index>
struct region_label<tf::preserve_regions_t<Range>, Index> {
  using type = typename Range::value_type;
};

template <typename Regions, typename Index>
using region_label_t = typename region_label<Regions, Index>::type;

} // namespace tf::remesh
