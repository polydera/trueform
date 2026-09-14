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

// The compiled build tier for double coordinates, int32 and int64 indices.
#include "graph_builders_impl.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::cpp::graph_builders {
namespace {
template <std::size_t Ngon> using a = form_t<std::int32_t, double, Ngon>;
template <std::size_t Ngon> using b = form_t<std::int64_t, double, Ngon>;
} // namespace

#define TF_CPP_INSTANTIATE_PAIR_BUILDERS(Ngon0, Ngon1)                         \
  template pair_arrangement_t<a<Ngon0>, b<Ngon1>>                              \
  build_pair_arrangement<a<Ngon0>, b<Ngon1>>(                                  \
      const a<Ngon0> &, const b<Ngon1> &, tf::arrangement_config);             \
  template pair_csg_graph_t<a<Ngon0>, b<Ngon1>>                                \
  build_pair_csg_graph<a<Ngon0>, b<Ngon1>>(const a<Ngon0> &, const b<Ngon1> &, \
                                           sheets_t, tf::arrangement_config)

TF_CPP_MATRIX_FOR_EACH_NGON_PAIR(TF_CPP_INSTANTIATE_PAIR_BUILDERS)

#undef TF_CPP_INSTANTIATE_PAIR_BUILDERS

} // namespace tf::cpp::graph_builders
