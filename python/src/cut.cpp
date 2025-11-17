/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/cut.hpp"

namespace tf::py {

auto register_cut(nanobind::module_ &m) -> void {
  // Register cut components
  register_cut_isobands(m);
  register_cut_boolean(m);
}

} // namespace tf::py
