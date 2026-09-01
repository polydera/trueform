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
// The compiled build tier of the single-operand entries for int64,
// float64, 3D: the self arrangement and the N-operand arrangement.
#include "trueform/python/arrangement/arrangement_builders_impl.hpp"

#include <cstdint>

namespace tf::py {
namespace {
using tri = form_t<std::int64_t, double, 3, 3>;
using dyn = form_t<std::int64_t, double, tf::dynamic_size, 3>;
using tri_forms = forms_range_t<std::int64_t, double, 3, 3>;
using dyn_forms = forms_range_t<std::int64_t, double, tf::dynamic_size, 3>;
} // namespace

template self_arrangement_t<tri>
build_self_arrangement<tri>(const tri &, tf::arrangement_config);
template self_arrangement_t<dyn>
build_self_arrangement<dyn>(const dyn &, tf::arrangement_config);

template range_arrangement_t<tri_forms>
build_range_arrangement<tri_forms>(tri_forms, tf::arrangement_config);
template range_arrangement_t<dyn_forms>
build_range_arrangement<dyn_forms>(dyn_forms, tf::arrangement_config);

} // namespace tf::py
