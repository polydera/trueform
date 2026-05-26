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
#include <nanobind/stl/tuple.h>

#include <trueform/python/core/offset_blocked_array.hpp>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/topology/make_sidedness_relations.hpp>

namespace tf::py::impl {

template <typename Index, std::size_t Ngon, typename RealT, std::size_t Dims>
auto make_sidedness_relations_impl(
    mesh_wrapper<Index, RealT, Ngon, Dims> &wrapper,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>>
        tag_labels) {
  auto polygons = wrapper.make_primitive_range() |
                  tf::tag(wrapper.face_membership()) |
                  tf::tag(wrapper.manifold_edge_link());

  auto tag_range =
      tf::make_range(static_cast<const Index *>(tag_labels.data()),
                     tag_labels.size());

  auto [relations, cl] = tf::make_sidedness_relations(polygons, tag_range);

  auto n_entries = relations.data_buffer().size();
  tf::buffer<Index> tags_flat;
  tf::buffer<Index> sides_flat;
  tags_flat.allocate(n_entries);
  sides_flat.allocate(n_entries);
  for (std::size_t i = 0; i < n_entries; ++i) {
    auto &e = relations.data_buffer()[i];
    tags_flat[i] = e.tag;
    sides_flat[i] = static_cast<Index>(e.side);
  }

  auto offsets_np =
      make_numpy_array(std::move(relations.offsets_buffer()));
  auto tags_np = make_numpy_array(std::move(tags_flat));
  auto sides_np = make_numpy_array(std::move(sides_flat));
  auto labels_np = make_numpy_array(std::move(cl.labels));

  return nanobind::make_tuple(
      nanobind::make_tuple(
          offset_blocked_array_wrapper<Index, Index>{offsets_np, tags_np},
          offset_blocked_array_wrapper<Index, Index>{offsets_np, sides_np}),
      nanobind::make_tuple(static_cast<Index>(cl.n_components), labels_np));
}

} // namespace tf::py::impl
