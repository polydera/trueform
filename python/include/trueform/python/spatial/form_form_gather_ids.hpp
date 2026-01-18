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
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <string>
#include <trueform/core/distance.hpp>
#include <trueform/core/frame.hpp>
#include <trueform/core/intersects.hpp>
#include <trueform/core/policy/frame.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/core/form.hpp>
#include <trueform/spatial/policy/tree.hpp>
#include <trueform/spatial/gather_ids.hpp>
#include <type_traits>
#include <vector>

namespace tf::py {

template <typename RealT, std::size_t Dims, typename FormWrapper0,
          typename FormWrapper1>
auto form_form_gather_ids(FormWrapper0 &form_wrapper0,
                          FormWrapper1 &form_wrapper1,
                          const std::string &predicate_type,
                          std::optional<RealT> threshold) {
  using Index0 = typename std::decay_t<decltype(form_wrapper0.tree())>::index_type;
  using Index1 = typename std::decay_t<decltype(form_wrapper1.tree())>::index_type;
  using CommonIndex = std::common_type_t<Index0, Index1>;

  std::vector<std::pair<Index0, Index1>> buffer;

  bool has0 = form_wrapper0.has_transformation();
  bool has1 = form_wrapper1.has_transformation();

  auto form0 = form_wrapper0.make_primitive_range() | tf::tag(form_wrapper0.tree());

  auto form1 = form_wrapper1.make_primitive_range() | tf::tag(form_wrapper1.tree());
  if (predicate_type == "intersects") {
    auto aabbs_pred = [](const auto &aabb0, const auto &aabb1) {
      return tf::intersects(aabb0, aabb1);
    };
    auto prims_pred = [](const auto &obj0, const auto &obj1) {
      return tf::intersects(obj0, obj1);
    };

    if (has0 && has1) {
      tf::gather_ids(
          form0 | tf::tag(tf::make_frame(form_wrapper0.transformation_view())),
          form1 | tf::tag(tf::make_frame(form_wrapper1.transformation_view())),
          aabbs_pred, prims_pred, std::back_inserter(buffer));
    } else if (has0 && !has1) {
      tf::gather_ids(
          form0 | tf::tag(tf::make_frame(form_wrapper0.transformation_view())),
          form1, aabbs_pred, prims_pred, std::back_inserter(buffer));
    } else if (!has0 && has1) {
      tf::gather_ids(
          form0,
          form1 | tf::tag(tf::make_frame(form_wrapper1.transformation_view())),
          aabbs_pred, prims_pred, std::back_inserter(buffer));
    } else {
      tf::gather_ids(form0, form1, aabbs_pred, prims_pred,
                     std::back_inserter(buffer));
    }
  } else if (predicate_type == "within_distance") {
    if (!threshold)
      throw std::runtime_error("threshold required for within_distance");
    RealT threshold2 = (*threshold) * (*threshold);

    auto aabbs_pred = [threshold2](const auto &aabb0, const auto &aabb1) {
      return tf::distance2(aabb0, aabb1) <= threshold2;
    };
    auto prims_pred = [threshold2](const auto &obj0, const auto &obj1) {
      return tf::distance2(obj0, obj1) <= threshold2;
    };

    if (has0 && has1)
      tf::gather_ids(
          form0 | tf::tag(tf::make_frame(form_wrapper0.transformation_view())),
          form1 | tf::tag(tf::make_frame(form_wrapper1.transformation_view())),
          aabbs_pred, prims_pred, std::back_inserter(buffer));
    else if (has0 && !has1)
      tf::gather_ids(
          form0 | tf::tag(tf::make_frame(form_wrapper0.transformation_view())),
          form1, aabbs_pred, prims_pred, std::back_inserter(buffer));
    else if (!has0 && has1)
      tf::gather_ids(
          form0,
          form1 | tf::tag(tf::make_frame(form_wrapper1.transformation_view())),
          aabbs_pred, prims_pred, std::back_inserter(buffer));
    else
      tf::gather_ids(form0, form1, aabbs_pred, prims_pred,
                     std::back_inserter(buffer));
  } else {
    throw std::runtime_error("Unknown predicate: " + predicate_type);
  }

  // Convert to numpy array with common index type
  size_t n = buffer.size();
  CommonIndex *data = nullptr;

  if (n > 0) {
    data = new CommonIndex[n * 2];
    for (size_t i = 0; i < n; ++i) {
      data[i * 2] = static_cast<CommonIndex>(buffer[i].first);
      data[i * 2 + 1] = static_cast<CommonIndex>(buffer[i].second);
    }
  }

  auto result = make_numpy_array<nanobind::shape<-1, 2>>(data, {n, 2});
  return result;
}

} // namespace tf::py
