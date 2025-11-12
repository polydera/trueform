/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <trueform/core/frame.hpp>
#include <trueform/core/ray_like.hpp>
#include <trueform/spatial/form.hpp>
#include <trueform/spatial/ray_cast.hpp>

namespace tf::py {

template <std::size_t Dims, typename Policy, typename FormWrapper>
auto ray_cast(tf::ray_like<Dims, Policy> ray, FormWrapper &form_wrapper) {
  form_wrapper.ensure_tree();

  auto make_return = [](auto res) {
    using res_t = std::pair<decltype(res.element), decltype(res.info.t)>;
    if (res)
      return std::optional<res_t>(std::make_pair(res.element, res.info.t));
    else
      return std::optional<res_t>(std::nullopt);
  };

  if (form_wrapper.has_transformation()) {
    return make_return(tf::ray_cast(
        ray, tf::make_form(tf::make_frame(form_wrapper.transformation_view()),
                           form_wrapper.tree(),
                           form_wrapper.make_primitive_range())));
  } else {
    return make_return(
        tf::ray_cast(ray, tf::make_form(form_wrapper.tree(),
                                        form_wrapper.make_primitive_range())));
  }
}
} // namespace tf::py
