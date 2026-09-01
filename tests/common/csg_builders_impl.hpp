/**
 * @file csg_builders_impl.hpp
 * @brief The classification builders' bodies
 *
 * Included ONLY by a combination's builder translation unit. The
 * arrangement is not built here — it is the arrangement builder's
 * instantiation, called through its declaration.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#pragma once

#include "csg_builders.hpp"

#include <trueform/core/none.hpp>
#include <trueform/csg/make_csg_graph.hpp>

#include <utility>

namespace tf::test {

template <typename Form>
auto build_self_csg_graph(const Form &form, tf::arrangement_config config)
    -> self_csg_graph_t<Form> {
  return tf::csg::csg_graph_over<tf::none_t>(
      build_self_arrangement(form, config), no_sheets());
}

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

} // namespace tf::test
