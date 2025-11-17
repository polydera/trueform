/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/intersect.hpp"

namespace tf::py {

auto register_intersect(nanobind::module_ &m) -> void {
  // Register intersect components
  register_intersect_isocontours(m);
  register_intersect_intersection_curves(m);
}

} // namespace tf::py
