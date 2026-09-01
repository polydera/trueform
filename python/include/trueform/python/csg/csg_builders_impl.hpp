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
// The csg builders' bodies. Included ONLY by a combination's csg builder
// TU. The arrangement is not built here — it is the arrangement builder's
// instantiation, called through its declaration.
#include "./csg_builders.hpp"
#include <trueform/core/none.hpp>
#include <trueform/csg/make_csg_graph.hpp>
#include <utility>

namespace tf::py {

template <typename Form0, typename Form1>
auto build_pair_csg_graph(const Form0 &form0, const Form1 &form1,
                          tf::range<const int *, tf::dynamic_size> sheets,
                          tf::arrangement_config config)
    -> pair_csg_graph_t<Form0, Form1> {
  return tf::csg::csg_graph_over<tf::none_t>(
      build_pair_arrangement(form0, form1, config), sheets);
}

template <typename Forms>
auto build_range_csg_graph(Forms forms,
                           tf::range<const int *, tf::dynamic_size> sheets,
                           tf::arrangement_config config)
    -> range_csg_graph_t<Forms> {
  return tf::csg::csg_graph_over<tf::none_t>(
      build_range_arrangement(std::move(forms), config), sheets);
}

} // namespace tf::py
