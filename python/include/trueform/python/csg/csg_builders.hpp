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
// The classification tier over an arrangement the arrangement builder
// already produced. A csg builder TU calls the arrangement builder rather
// than re-instantiating the build, so a combination's graph is compiled
// exactly once no matter how many binding entries read it.
#include "../arrangement/arrangement_builders.hpp"
#include <trueform/core/none.hpp>
#include <trueform/core/range.hpp>
#include <trueform/csg/csg_graph.hpp>
#include <trueform/arrangement/arrangement_config.hpp>
#include <trueform/arrangement/arrangement_graph.hpp>
#include <trueform/arrangement/policy/arrangement_range_policy.hpp>

namespace tf::py {

template <typename Forms>
using range_csg_graph_t =
    tf::csg_graph<tf::arrangement::arrangement_range_policy<Forms, tf::none_t>,
                  tf::none_t, tf::arrangement_graph>;

/// N operands. The graph stores the range, so the forms behind it must
/// outlive the graph.
template <typename Forms>
auto build_range_csg_graph(Forms forms,
                           tf::range<const int *, tf::dynamic_size> sheets,
                           tf::arrangement_config config)
    -> range_csg_graph_t<Forms>;

} // namespace tf::py
