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
// The compiled build tier of the two-operand entries for int64 x
// int64, float64, 3D.
#include "trueform/python/arrangement/arrangement_builders_impl.hpp"

#include <cstdint>

namespace tf::py {
namespace {
using tri = form_t<std::int64_t, double, 3, 3>;
using dyn = form_t<std::int64_t, double, tf::dynamic_size, 3>;
} // namespace

template pair_arrangement_t<tri, tri>
build_pair_arrangement<tri, tri>(const tri &, const tri &,
                                 tf::arrangement_config);

template pair_arrangement_t<tri, dyn>
build_pair_arrangement<tri, dyn>(const tri &, const dyn &,
                                 tf::arrangement_config);

template pair_arrangement_t<dyn, tri>
build_pair_arrangement<dyn, tri>(const dyn &, const tri &,
                                 tf::arrangement_config);

template pair_arrangement_t<dyn, dyn>
build_pair_arrangement<dyn, dyn>(const dyn &, const dyn &,
                                 tf::arrangement_config);

} // namespace tf::py
