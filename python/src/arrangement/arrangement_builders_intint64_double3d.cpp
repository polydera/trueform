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
// The compiled build tier of the two-operand entries for int32 x
// int64, float64, 3D.
#include "trueform/python/arrangement/arrangement_builders_impl.hpp"

#include <cstdint>

namespace tf::py {
namespace {
using tri0 = form_t<int, double, 3, 3>;
using dyn0 = form_t<int, double, tf::dynamic_size, 3>;
using tri1 = form_t<std::int64_t, double, 3, 3>;
using dyn1 = form_t<std::int64_t, double, tf::dynamic_size, 3>;
} // namespace

template pair_arrangement_t<tri0, tri1>
build_pair_arrangement<tri0, tri1>(const tri0 &, const tri1 &,
                                   tf::arrangement_config);

template pair_arrangement_t<tri0, dyn1>
build_pair_arrangement<tri0, dyn1>(const tri0 &, const dyn1 &,
                                   tf::arrangement_config);

template pair_arrangement_t<dyn0, tri1>
build_pair_arrangement<dyn0, tri1>(const dyn0 &, const tri1 &,
                                   tf::arrangement_config);

template pair_arrangement_t<dyn0, dyn1>
build_pair_arrangement<dyn0, dyn1>(const dyn0 &, const dyn1 &,
                                   tf::arrangement_config);

} // namespace tf::py
