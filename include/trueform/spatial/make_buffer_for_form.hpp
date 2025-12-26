/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (www.trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/
#pragma once
#include "../core/make_buffer_for_transformed.hpp"
#include "../core/make_local_buffer_for_transformed.hpp"
#include "./form.hpp"

namespace tf::spatial {

template <std::size_t Dims, typename Policy>
auto make_buffer_for_form(const tf::form<Dims, Policy> &form) {
  return decltype(tf::core::make_buffer_for_transformed(form[0], form.frame())){};
}

template <std::size_t Dims, typename Policy>
auto make_local_buffer_for_form(const tf::form<Dims, Policy> &form) {
  return decltype(tf::core::make_local_buffer_for_transformed(form[0], form.frame())){};
}

} // namespace tf::spatial
