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

namespace tf::topology::cdt {

struct original_input_vertex_policy {
  template <typename Index, typename Site>
  static auto output(Index, const Site &site) -> Index {
    return site.output;
  }
};

struct compact_topology_vertex_policy {
  template <typename Index, typename Site>
  static auto output(Index topology_vertex, const Site &) -> Index {
    return topology_vertex;
  }
};

} // namespace tf::topology::cdt
