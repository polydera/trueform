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
// The compiled classification tier for int64, float32,
// 3D. Its bodies call the arrangement builders; the build tier is NOT
// re-instantiated here.
#include "trueform/python/csg/csg_builders_impl.hpp"

#include <cstdint>

namespace tf::py {
namespace {
using tri_forms = forms_range_t<std::int64_t, float, 3, 3>;
using dyn_forms = forms_range_t<std::int64_t, float, tf::dynamic_size, 3>;
} // namespace

template range_csg_graph_t<tri_forms>
build_range_csg_graph<tri_forms>(
    tri_forms, tf::range<const int *, tf::dynamic_size>,
    tf::arrangement_config);

template range_csg_graph_t<dyn_forms>
build_range_csg_graph<dyn_forms>(
    dyn_forms, tf::range<const int *, tf::dynamic_size>,
    tf::arrangement_config);

} // namespace tf::py
