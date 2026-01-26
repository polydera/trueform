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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <trueform/core/frame.hpp>
#include <trueform/core/ray_config.hpp>
#include <trueform/core/ray_like.hpp>
#include <trueform/python/util/ray_config_helper.hpp>
#include <trueform/core/form.hpp>
#include <trueform/spatial/policy/tree.hpp>
#include <trueform/spatial/ray_cast.hpp>
#include <tuple>

namespace tf::py {

template <std::size_t Dims, typename Policy, typename FormWrapper>
auto ray_cast(tf::ray_like<Dims, Policy> ray, FormWrapper &form_wrapper,
              std::optional<std::tuple<tf::coordinate_type<Policy>,
                                       tf::coordinate_type<Policy>>>
                  opt_config = std::nullopt) {
  // Create ray_config from optional tuple
  auto config = make_ray_config_from_optional(opt_config);

  auto make_return = [](auto res) {
    using res_t = std::pair<decltype(res.element), decltype(res.info.t)>;
    if (res)
      return std::optional<res_t>(std::make_pair(res.element, res.info.t));
    else
      return std::optional<res_t>(std::nullopt);
  };

  if (form_wrapper.has_transformation()) {
    return make_return(tf::ray_cast(
        ray,
        form_wrapper.make_primitive_range() | tf::tag(form_wrapper.tree()) |
            tf::tag(tf::make_frame(form_wrapper.transformation_view())),
        config));
  } else {
    return make_return(tf::ray_cast(
        ray,
        form_wrapper.make_primitive_range() | tf::tag(form_wrapper.tree()),
        config));
  }
}
} // namespace tf::py
