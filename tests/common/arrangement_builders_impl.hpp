/**
 * @file arrangement_builders_impl.hpp
 * @brief The arrangement builders' bodies
 *
 * Included ONLY by a combination's builder translation unit, which
 * explicitly instantiates the forms it owns; a test TU includes the
 * declarations and links against those instantiations.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#pragma once

#include "arrangement_builders.hpp"

#include <trueform/arrangement/make_arrangement_graph.hpp>
#include <trueform/core/none.hpp>

#include <utility>

namespace tf::test {

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

} // namespace tf::test
