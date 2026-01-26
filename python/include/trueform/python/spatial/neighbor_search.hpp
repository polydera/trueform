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
#include <nanobind/stl/vector.h>
#include <optional>
#include <trueform/core/frame.hpp>
#include <trueform/core/form.hpp>
#include <trueform/spatial/nearest_neighbor.hpp>
#include <trueform/spatial/policy/tree.hpp>
#include <trueform/spatial/neighbor_search.hpp>
#include <vector>

namespace tf::py {

template <typename RealT, std::size_t Dims, typename FormWrapper,
          typename Primitive>
auto neighbor_search(FormWrapper &form_wrapper, const Primitive &query,
                     std::optional<RealT> radius) {
  using ndarray_t =
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<Dims>>;
  using Index = typename std::decay_t<decltype(form_wrapper.tree())>::index_type;
  using result_t = std::tuple<Index, RealT, ndarray_t>;

  auto make_return = [](const auto &e) -> std::optional<result_t> {
    if (!e)
      return std::nullopt;
    // Allocate numpy array
    RealT *data = new RealT[Dims];
    const auto &pt = e.info.point;
    std::copy(pt.begin(), pt.end(), data);

    // Create ndarray with ownership via capsule
    auto capsule = nanobind::capsule(
        data, [](void *p) noexcept { delete[] static_cast<RealT *>(p); });

    ndarray_t arr(data, {Dims}, capsule);
    return result_t{e.element, e.info.metric, arr};
  };

  RealT r = std::numeric_limits<RealT>::max();
  if (radius)
    r = *radius;
  if (form_wrapper.has_transformation()) {
    return make_return(tf::neighbor_search(
        form_wrapper.make_primitive_range() | tf::tag(form_wrapper.tree()) |
            tf::tag(tf::make_frame(form_wrapper.transformation_view())),
        query, r));
  } else {
    return make_return(tf::neighbor_search(
        form_wrapper.make_primitive_range() | tf::tag(form_wrapper.tree()),
        query, r));
  }
}

template <typename RealT, std::size_t Dims, typename FormWrapper,
          typename Primitive>
auto neighbor_search(FormWrapper &from_wrapper, const Primitive &query, int k,
                     std::optional<RealT> radius) {
  using Index = typename std::decay_t<decltype(from_wrapper.tree())>::index_type;
  using ndarray_t =
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<Dims>>;
  using result_t = std::tuple<int, RealT, ndarray_t>;
  std::vector<result_t> results;

  RealT r = std::numeric_limits<RealT>::max();
  if (radius)
    r = *radius;

  std::vector<tf::nearest_neighbor<Index, RealT, Dims>> knn_buffer(k);
  auto knn = tf::make_nearest_neighbors(knn_buffer.begin(), k, r);

  if (from_wrapper.has_transformation()) {
    tf::neighbor_search(
        from_wrapper.make_primitive_range() | tf::tag(from_wrapper.tree()) |
            tf::tag(tf::make_frame(from_wrapper.transformation_view())),
        query, knn);
  } else {
    tf::neighbor_search(
        from_wrapper.make_primitive_range() | tf::tag(from_wrapper.tree()),
        query, knn);
  }

  results.reserve(knn.size());
  for (const auto &e : knn) {
    // Allocate numpy array
    RealT *data = new RealT[Dims];
    const auto &pt = e.info.point;
    std::copy(pt.begin(), pt.end(), data);

    // Create ndarray with ownership via capsule
    auto capsule = nanobind::capsule(
        data, [](void *p) noexcept { delete[] static_cast<RealT *>(p); });

    ndarray_t arr(data, {Dims}, capsule);
    results.emplace_back(e.element, e.info.metric, arr);
  }

  return results;
}

} // namespace tf::py
