/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <trueform/trueform.hpp>
#include <vector>

namespace tf::py {

template <typename RealT, std::size_t Dims, typename FormWrapper,
          typename Primitive>
auto neighbor_search(FormWrapper &form_wrapper, const Primitive &query,
                     std::optional<RealT> radius) {

  using result_t = std::tuple<int, RealT, std::array<RealT, Dims>>;
  auto make_return = [](const auto &e) -> result_t {
    std::array<RealT, Dims> outr;
    const auto &pt = e.info.point;
    std::copy(pt.begin(), pt.end(), outr.begin());
    return {e.element, e.info.metric, outr};
  };
  form_wrapper.ensure_tree();

  RealT r = std::numeric_limits<RealT>::max();
  if (radius)
    r = *radius;
  if (form_wrapper.has_transformation()) {
    return make_return(tf::neighbor_search(
        tf::make_form(tf::make_frame(form_wrapper.transformation_view()),
                      form_wrapper.tree(), form_wrapper.make_primitive_range()),
        query, r));
  } else {
    return make_return(tf::neighbor_search(
        tf::make_form(form_wrapper.tree(), form_wrapper.make_primitive_range()),
        query, r));
  }
}

template <typename RealT, std::size_t Dims, typename FormWrapper,
          typename Primitive>
auto neighbor_search(FormWrapper &from_wrapper, const Primitive &query, int k,
                     std::optional<RealT> radius) {
  using result_t = std::tuple<int, RealT, std::array<RealT, Dims>>;
  std::vector<result_t> results;

  RealT r = std::numeric_limits<RealT>::max();
  if (radius)
    r = *radius;

  std::vector<tf::nearest_neighbor<int, RealT, Dims>> knn_buffer(k);
  auto knn = tf::make_nearest_neighbors(knn_buffer.begin(), k, r);

  if (from_wrapper.has_transformation()) {
    tf::neighbor_search(
        tf::make_form(tf::make_frame(from_wrapper.transformation_view()),
                      from_wrapper.tree(), from_wrapper.make_primitive_range()),
        query, knn);
  } else {
    tf::neighbor_search(
        tf::make_form(from_wrapper.tree(), from_wrapper.make_primitive_range()),
        query, knn);
  }

  results.reserve(knn.size());
  for (const auto &e : knn) {
    std::array<RealT, Dims> outr;
    const auto &pt = e.info.point;
    std::copy(pt.begin(), pt.end(), outr.begin());
    results.emplace_back(e.element, e.info.metric, outr);
  }

  return results;
}

} // namespace tf::py
