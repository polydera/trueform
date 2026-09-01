/**
 * @file arrangement_builders.hpp
 * @brief The suite's compiled build tier for the arrangement
 *
 * Every arrangement entry shares one prefix — the graph build — so the
 * graph type is NAMED here structurally and BUILT in one translation unit
 * per combination. A test TU includes these declarations and compiles a
 * call; it instantiates no build kernel.
 *
 * A builder is always declared with an explicit structural return type: a
 * deduced return would instantiate the factory body in every consumer,
 * which is exactly the cost this tier exists to move.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#pragma once

#include "tagged_operand.hpp"

#include <trueform/arrangement/arrangement_config.hpp>
#include <trueform/arrangement/arrangement_graph.hpp>
#include <trueform/arrangement/policy/arrangement_pair_policy.hpp>
#include <trueform/arrangement/policy/arrangement_range_policy.hpp>
#include <trueform/core/none.hpp>

#include <array>

namespace tf::test {

// A tagged operand carries tree, face membership and manifold edge link,
// so the build completes no structure and the graph's structs type is
// tf::none_t. A builder TU proves it: its definition returns whatever
// tf::make_arrangement_graph deduced, against the type declared here.
template <typename Form>
using self_arrangement_t = tf::arrangement_graph<
    tf::arrangement::arrangement_range_policy<std::array<Form, 1>, tf::none_t>,
    tf::none_t>;

template <typename Form0, typename Form1>
using pair_arrangement_t =
    tf::arrangement_graph<tf::arrangement::arrangement_pair_policy<
                              Form0, tf::none_t, Form1, tf::none_t>,
                          tf::none_t>;

template <typename Forms>
using range_arrangement_t = tf::arrangement_graph<
    tf::arrangement::arrangement_range_policy<Forms, tf::none_t>, tf::none_t>;

/// The form's self arrangement (`within` implied).
template <typename Form>
auto build_self_arrangement(const Form &form, tf::arrangement_config config)
    -> self_arrangement_t<Form>;

/// Two operands, possibly of different arity.
template <typename Form0, typename Form1>
auto build_pair_arrangement(const Form0 &form0, const Form1 &form1,
                            tf::arrangement_config config)
    -> pair_arrangement_t<Form0, Form1>;

/// N operands. The graph stores the range, so the forms behind it must
/// outlive the graph.
template <typename Forms>
auto build_range_arrangement(Forms forms, tf::arrangement_config config)
    -> range_arrangement_t<Forms>;

} // namespace tf::test
