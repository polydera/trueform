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

#include "trueform/python/csg/csg_graph_impl.hpp"

namespace tf::py {

auto register_csg_graph_int643double3d(nanobind::module_ &m) -> void {
  register_csg_graph<std::int64_t, double>(m, "CsgGraph_int643double3d");
}

} // namespace tf::py
