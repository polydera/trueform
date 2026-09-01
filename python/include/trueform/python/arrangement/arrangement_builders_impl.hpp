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
// The builders' bodies. Included ONLY by a combination's builder TU,
// which explicitly instantiates the forms it owns; a consumer TU includes
// the declarations and links against those instantiations.
#include "./arrangement_builders.hpp"
#include <trueform/core/none.hpp>
#include <trueform/arrangement/make_arrangement_graph.hpp>
#include <utility>

namespace tf::py {

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

} // namespace tf::py
