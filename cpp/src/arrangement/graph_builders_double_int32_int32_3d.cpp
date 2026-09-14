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

// The compiled build tier for double coordinates, int32 and int32 indices.
#include "graph_builders_impl.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::cpp::graph_builders {
namespace {
template <std::size_t Ngon> using a = form_t<std::int32_t, double, Ngon>;
template <std::size_t Ngon>
using a_forms = forms_range_t<std::int32_t, double, Ngon>;
} // namespace

#define TF_CPP_INSTANTIATE_PAIR_BUILDERS(Ngon0, Ngon1)                         \
  template pair_arrangement_t<a<Ngon0>, a<Ngon1>>                              \
  build_pair_arrangement<a<Ngon0>, a<Ngon1>>(                                  \
      const a<Ngon0> &, const a<Ngon1> &, tf::arrangement_config);             \
  template pair_csg_graph_t<a<Ngon0>, a<Ngon1>>                                \
  build_pair_csg_graph<a<Ngon0>, a<Ngon1>>(const a<Ngon0> &, const a<Ngon1> &, \
                                           sheets_t, tf::arrangement_config)

#define TF_CPP_INSTANTIATE_SELF_BUILDERS(Ngon)                                 \
  template self_arrangement_t<a<Ngon>> build_self_arrangement<a<Ngon>>(        \
      const a<Ngon> &, tf::arrangement_config);                                \
  template self_csg_graph_t<a<Ngon>> build_self_csg_graph<a<Ngon>>(            \
      const a<Ngon> &, tf::arrangement_config)

#define TF_CPP_INSTANTIATE_RANGE_BUILDERS(Ngon)                                \
  template range_arrangement_t<a_forms<Ngon>>                                  \
  build_range_arrangement<a_forms<Ngon>>(a_forms<Ngon>,                        \
                                         tf::arrangement_config)

TF_CPP_MATRIX_FOR_EACH_NGON_PAIR(TF_CPP_INSTANTIATE_PAIR_BUILDERS)
TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_SELF_BUILDERS)
TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_RANGE_BUILDERS)

#undef TF_CPP_INSTANTIATE_RANGE_BUILDERS
#undef TF_CPP_INSTANTIATE_SELF_BUILDERS
#undef TF_CPP_INSTANTIATE_PAIR_BUILDERS

/// A csg arrangement of a range is triangles, so its builder states one arity.
template range_csg_graph_t<a_forms<3>>
    build_range_csg_graph<a_forms<3>>(a_forms<3>, sheets_t,
                                      tf::arrangement_config);

} // namespace tf::cpp::graph_builders
