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

#include <algorithm>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <trueform/core/blocked_buffer.hpp>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/python/core/offset_blocked_array.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/reindex/split_into_domains.hpp>
#include <trueform/topology/domain_labels.hpp>

namespace tf::py {
namespace impl {

template <typename Index>
auto build_domain_labels(
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1, 2>>
        labels_flat,
    Index n_domains) -> tf::domain_labels<Index> {
  tf::domain_labels<Index> dl;
  dl.n_domains = n_domains;
  // outer_shell_label is unused by tf::split_into_domains — leave default.
  auto n_faces = static_cast<std::size_t>(labels_flat.shape(0));
  dl.labels.allocate(n_faces);
  auto &flat = dl.labels.data_buffer();
  std::copy(labels_flat.data(), labels_flat.data() + 2 * n_faces, flat.begin());
  return dl;
}

template <typename Components>
auto pack_components(Components &&components) {
  auto make_pair = [&components](auto i) {
    auto [indices, pts] = make_numpy_array(std::move(components[i]));
    return nanobind::make_tuple(std::move(indices), std::move(pts));
  };
  std::vector<decltype(make_pair(0))> cs;
  cs.reserve(components.size());
  for (std::size_t i = 0; i < components.size(); ++i)
    cs.push_back(make_pair(i));
  return cs;
}

template <typename Index, std::size_t V, typename RealT, std::size_t Dims>
auto split_into_domains_impl(
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1, V>>
        indices,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        points,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1, 2>>
        domain_labels_flat,
    Index n_domains) {
  auto i_range =
      tf::make_blocked_range<V>(tf::make_range(indices.data(), indices.size()));
  auto p_range =
      tf::make_points<Dims>(tf::make_range(points.data(), points.size()));
  auto polygons = tf::make_polygons(i_range, p_range);

  auto dl = build_domain_labels<Index>(domain_labels_flat, n_domains);
  auto [components, comp_labels] = tf::split_into_domains(polygons, dl);

  auto cs = pack_components(std::move(components));
  return nanobind::make_tuple(std::move(cs),
                              make_numpy_array(std::move(comp_labels)));
}

template <typename Index, typename RealT, std::size_t Dims>
auto split_into_domains_impl_dynamic(
    const offset_blocked_array_wrapper<Index, Index> &indices,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        points,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1, 2>>
        domain_labels_flat,
    Index n_domains) {
  auto p_range =
      tf::make_points<Dims>(tf::make_range(points.data(), points.size()));
  auto i_range = indices.make_range();
  auto polygons = tf::make_polygons(i_range, p_range);

  auto dl = build_domain_labels<Index>(domain_labels_flat, n_domains);
  auto [components, comp_labels] = tf::split_into_domains(polygons, dl);

  auto cs = pack_components(std::move(components));
  return nanobind::make_tuple(std::move(cs),
                              make_numpy_array(std::move(comp_labels)));
}

} // namespace impl
} // namespace tf::py
