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
// The builders' bodies. Included ONLY by a combination's builder translation
// unit, which explicitly instantiates the forms it owns; a consumer shard
// includes the declarations and links against those instantiations.

#include "graph_builders.hpp"

#include "trueform/arrangement/make_arrangement_graph.hpp"
#include "trueform/core/none.hpp"
#include "trueform/csg/make_csg_graph.hpp"

#include <utility>

namespace tf::cpp::graph_builders {

template <typename Form>
auto build_self_arrangement(const Form &form, tf::arrangement_config config)
    -> self_arrangement_t<Form> {
  return tf::make_arrangement_graph<tf::none_t>(form, config);
}

template <typename Form0, typename Form1>
auto build_pair_arrangement(const Form0 &form0, const Form1 &form1,
                            tf::arrangement_config config)
    -> pair_arrangement_t<Form0, Form1> {
  return tf::make_arrangement_graph<tf::none_t>(form0, form1, config);
}

template <typename Forms>
auto build_range_arrangement(Forms forms, tf::arrangement_config config)
    -> range_arrangement_t<Forms> {
  return tf::make_arrangement_graph<tf::none_t>(std::move(forms), config);
}

template <typename Form>
auto build_self_csg_graph(const Form &form, tf::arrangement_config config)
    -> self_csg_graph_t<Form> {
  return tf::csg::csg_graph_over<tf::none_t>(
      build_self_arrangement(form, config), no_sheets());
}

template <typename Form0, typename Form1>
auto build_pair_csg_graph(const Form0 &form0, const Form1 &form1,
                          sheets_t sheets, tf::arrangement_config config)
    -> pair_csg_graph_t<Form0, Form1> {
  return tf::csg::csg_graph_over<tf::none_t>(
      build_pair_arrangement(form0, form1, config), sheets);
}

template <typename Forms>
auto build_range_csg_graph(Forms forms, sheets_t sheets,
                           tf::arrangement_config config)
    -> range_csg_graph_t<Forms> {
  return tf::csg::csg_graph_over<tf::none_t>(
      build_range_arrangement(std::move(forms), config), sheets);
}

} // namespace tf::cpp::graph_builders
