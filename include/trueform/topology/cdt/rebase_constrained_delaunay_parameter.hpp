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
#include "../../exact/meta.hpp"
#include "../../exact/rebase_parameter.hpp"

namespace tf::topology::cdt {

/// Recovery recurses over constraint pieces, but provenance and crossing
/// reports name the whole physical input edge. Rebase a piece-local parameter
/// through its oriented interval before carrying it outward.
template <typename Owner>
auto rebase_constrained_delaunay_parameter(typename Owner::param_type local,
                                           typename Owner::param_type first,
                                           typename Owner::param_type second) ->
    typename Owner::param_type {
  using Int = typename Owner::int_type;
  return tf::exact::rebase_parameter<Int>(local, first, second,
                                          typename tf::exact::meta<Int>::T2(1)
                                              << Owner::crossing_param_bits);
}

} // namespace tf::topology::cdt
