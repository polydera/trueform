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

#include <trueform/arrangement/identity/near_identity_request.hpp>
#include <trueform/arrangement/operands/make_completed_graph.hpp>
#include <trueform/core/none.hpp>

#include <utility>

namespace tf::test {

template <typename Form>
auto build_self_arrangement(const Form &form, tf::arrangement_config config,
                            tf::arrangement::near_identity_request request)
    -> self_arrangement_t<Form> {
  return tf::arrangement::dispatch::make_completed_graph<tf::none_t>(
      form, config, request);
}

template <typename Form0, typename Form1>
auto build_pair_arrangement(const Form0 &form0, const Form1 &form1,
                            tf::arrangement_config config,
                            tf::arrangement::near_identity_request request)
    -> pair_arrangement_t<Form0, Form1> {
  return tf::arrangement::dispatch::make_completed_graph<tf::none_t>(
      form0, form1, config, request);
}

template <typename Forms>
auto build_range_arrangement(Forms forms, tf::arrangement_config config,
                             tf::arrangement::near_identity_request request)
    -> range_arrangement_t<Forms> {
  return tf::arrangement::dispatch::make_completed_graph<tf::none_t>(
      std::move(forms), config, request);
}

} // namespace tf::test
