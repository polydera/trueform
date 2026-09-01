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
// The compiled classification tier for int32 x int32, float32,
// 3D. Its bodies call the arrangement builders; the build tier is NOT
// re-instantiated here.
#include "trueform/python/csg/csg_builders_impl.hpp"

#include <cstdint>

namespace tf::py {
namespace {
using tri = form_t<int, float, 3, 3>;
using dyn = form_t<int, float, tf::dynamic_size, 3>;
using tri_forms = forms_range_t<int, float, 3, 3>;
using dyn_forms = forms_range_t<int, float, tf::dynamic_size, 3>;
} // namespace

template pair_csg_graph_t<tri, tri>
build_pair_csg_graph<tri, tri>(const tri &, const tri &,
                               tf::range<const int *, tf::dynamic_size>,
                               tf::arrangement_config);

template pair_csg_graph_t<tri, dyn>
build_pair_csg_graph<tri, dyn>(const tri &, const dyn &,
                               tf::range<const int *, tf::dynamic_size>,
                               tf::arrangement_config);

template pair_csg_graph_t<dyn, tri>
build_pair_csg_graph<dyn, tri>(const dyn &, const tri &,
                               tf::range<const int *, tf::dynamic_size>,
                               tf::arrangement_config);

template pair_csg_graph_t<dyn, dyn>
build_pair_csg_graph<dyn, dyn>(const dyn &, const dyn &,
                               tf::range<const int *, tf::dynamic_size>,
                               tf::arrangement_config);

template range_csg_graph_t<tri_forms>
build_range_csg_graph<tri_forms>(
    tri_forms, tf::range<const int *, tf::dynamic_size>,
    tf::arrangement_config);

template range_csg_graph_t<dyn_forms>
build_range_csg_graph<dyn_forms>(
    dyn_forms, tf::range<const int *, tf::dynamic_size>,
    tf::arrangement_config);

} // namespace tf::py
