/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/topology.hpp"

namespace tf::py {

auto register_topology(nanobind::module_ &m) -> void {
  // Create topology submodule
  auto topology_module = m.def_submodule("topology", "Topology operations");

  // Register topology components to submodule
  register_topology_label_connected_components(topology_module);
}

} // namespace tf::py
