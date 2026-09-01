/**
 * @file csg_builders.hpp
 * @brief The suite's compiled classification tier
 *
 * The classification over an arrangement the arrangement builder already
 * produced. A csg builder TU calls the arrangement builder rather than
 * re-instantiating the build, so a combination's graph is compiled exactly
 * once no matter how many test TUs read it.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#pragma once

#include "arrangement_builders.hpp"

#include <trueform/arrangement/arrangement_config.hpp>
#include <trueform/arrangement/arrangement_graph.hpp>
#include <trueform/arrangement/policy/arrangement_pair_policy.hpp>
#include <trueform/arrangement/policy/arrangement_range_policy.hpp>
#include <trueform/core/none.hpp>
#include <trueform/core/range.hpp>
#include <trueform/csg/csg_graph.hpp>

#include <array>
#include <vector>

namespace tf::test {

/// What a test passes when every operand bounds a volume.
inline auto no_sheets() -> tf::range<const int *, tf::dynamic_size> {
  const int *none = nullptr;
  return tf::make_range(none, none);
}

/// The sheet operands, in the one range type the builders are compiled on.
inline auto sheets_of(const std::vector<int> &ids)
    -> tf::range<const int *, tf::dynamic_size> {
  const int *first = ids.data();
  return tf::make_range(first, first + ids.size());
}

template <typename Form>
using self_csg_graph_t =
    tf::csg_graph<tf::arrangement::arrangement_range_policy<std::array<Form, 1>,
                                                            tf::none_t>,
                  tf::none_t, tf::arrangement_graph>;

template <typename Form0, typename Form1>
using pair_csg_graph_t =
    tf::csg_graph<tf::arrangement::arrangement_pair_policy<Form0, tf::none_t,
                                                           Form1, tf::none_t>,
                  tf::none_t, tf::arrangement_graph>;

template <typename Forms>
using range_csg_graph_t =
    tf::csg_graph<tf::arrangement::arrangement_range_policy<Forms, tf::none_t>,
                  tf::none_t, tf::arrangement_graph>;

/// One operand: the form's self arrangement classified.
template <typename Form>
auto build_self_csg_graph(const Form &form, tf::arrangement_config config)
    -> self_csg_graph_t<Form>;

/// Two operands, possibly of different arity.
template <typename Form0, typename Form1>
auto build_pair_csg_graph(const Form0 &form0, const Form1 &form1,
                          tf::range<const int *, tf::dynamic_size> sheets,
                          tf::arrangement_config config)
    -> pair_csg_graph_t<Form0, Form1>;

/// N operands. The graph stores the range, so the forms behind it must
/// outlive the graph.
template <typename Forms>
auto build_range_csg_graph(Forms forms,
                           tf::range<const int *, tf::dynamic_size> sheets,
                           tf::arrangement_config config)
    -> range_csg_graph_t<Forms>;

} // namespace tf::test
