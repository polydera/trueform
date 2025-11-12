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
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <trueform/core/coordinate_dims.hpp>
#include <trueform/core/coordinate_type.hpp>
#include <trueform/core/frame.hpp>
#include <trueform/core/policy/frame.hpp>
#include <trueform/spatial/form.hpp>
#include <trueform/spatial/neighbor_search.hpp>

namespace tf::py {
template <typename FormWrapper0, typename FormWrapper1, typename RealT>
auto form_form_neighbor_search(FormWrapper0 &form_wrapper0,
                               FormWrapper1 &form_wrapper1,
                               std::optional<RealT> radius) {

  form_wrapper0.ensure_tree();
  form_wrapper1.ensure_tree();
  bool has0 = form_wrapper0.has_transformation();
  bool has1 = form_wrapper1.has_transformation();
  auto form0 =
      tf::make_form(form_wrapper0.tree(), form_wrapper0.make_primitive_range());
  auto form1 =
      tf::make_form(form_wrapper1.tree(), form_wrapper1.make_primitive_range());

  constexpr auto Dims = tf::coordinate_dims_v<decltype(form0)>;
  using Index0 = tf::coordinate_type<decltype(form0)>;
  using Index1 = tf::coordinate_type<decltype(form0)>;

  using ndarray_t =
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<Dims>>;
  using result_t = std::tuple<std::pair<Index0, Index1>,
                              std::tuple<RealT, ndarray_t, ndarray_t>>;

  auto make_return = [](const auto &e) -> std::optional<result_t> {
    if (!e)
      return std::nullopt;
    auto make_point = [](const auto &pt) {
      // Allocate numpy array
      RealT *data = new RealT[Dims];
      std::copy(pt.begin(), pt.end(), data);

      // Create ndarray with ownership via capsule
      auto capsule = nanobind::capsule(
          data, [](void *p) noexcept { delete[] static_cast<RealT *>(p); });

      ndarray_t arr(data, {Dims}, capsule);
      return arr;
    };
    return result_t{
        e.elements,
        {e.info.metric, make_point(e.info.first), make_point(e.info.second)}};
  };

  RealT r = std::numeric_limits<RealT>::max();
  if (radius)
    r = *radius;

  if (has0 && has1)
    return make_return(tf::neighbor_search(
        form0 | tf::tag(tf::make_frame(form_wrapper0.transformation_view())),
        form1 | tf::tag(tf::make_frame(form_wrapper1.transformation_view())),
        r));
  else if (has0 && !has1)
    return make_return(tf::neighbor_search(
        form0 | tf::tag(tf::make_frame(form_wrapper0.transformation_view())),
        form1, r));
  else if (!has0 && has1)
    return make_return(tf::neighbor_search(
        form0,
        form1 | tf::tag(tf::make_frame(form_wrapper1.transformation_view())),
        r));
  else
    return make_return(tf::neighbor_search(form0, form1, r));
}
} // namespace tf::py
