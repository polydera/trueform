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
#include "../intersect/build_intersect_structures.hpp"
#include "../spatial/mesh.hpp"
#include "../util/make_numpy_array.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <trueform/core/transformation.hpp>
#include <trueform/cut/arrangement_config.hpp>
#include <trueform/cut/make_mesh_arrangements.hpp>
#include <trueform/topology/triangulation_type.hpp>

namespace tf::py {

template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto mesh_arrangements(
    std::vector<mesh_wrapper<Index, RealT, Ngon, Dims>> &wrappers, int mode,
    double tolerance, int triangulation) {
  bool any_transformed = false;
  for (auto &w : wrappers)
    if (w.has_transformation())
      any_transformed = true;

  build_intersect_structures_all(wrappers);

  auto run = [mode, tolerance, triangulation](const auto &forms) {
    auto [mesh, tag_labels, face_labels] = tf::make_mesh_arrangements(
        forms,
        tf::arrangement_config{
            tf::intersect_config{static_cast<tf::intersect_mode>(mode),
                                 tolerance},
            static_cast<tf::triangulation_type>(triangulation)});
    return nanobind::make_tuple(make_numpy_array(std::move(mesh)),
                                make_numpy_array(std::move(tag_labels)),
                                make_numpy_array(std::move(face_labels)));
  };

  auto range = tf::make_range(wrappers.data(), wrappers.size());

  if (any_transformed) {
    auto identity = tf::make_identity_transformation<RealT, Dims>();
    auto tagged = tf::make_mapped_range(range, [&](auto &w) {
      auto fm = w.face_membership();
      auto mel = w.manifold_edge_link();
      tf::transformation<RealT, Dims> tv = w.has_transformation()
          ? tf::transformation<RealT, Dims>(w.transformation_view())
          : identity;
      return w.make_primitive_range() | tf::tag(w.tree()) | tf::tag(fm) |
             tf::tag(mel) | tf::tag(tv);
    });
    return run(tagged);
  }

  auto tagged = tf::make_mapped_range(range, [](auto &w) {
    auto fm = w.face_membership();
    auto mel = w.manifold_edge_link();
    return w.make_primitive_range() | tf::tag(w.tree()) | tf::tag(fm) |
           tf::tag(mel);
  });
  return run(tagged);
}

template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto mesh_arrangements(
    std::vector<mesh_wrapper<Index, RealT, Ngon, Dims>> &wrappers, int mode,
    double tolerance, int triangulation, tf::return_curves_t) {
  bool any_transformed = false;
  for (auto &w : wrappers)
    if (w.has_transformation())
      any_transformed = true;

  build_intersect_structures_all(wrappers);

  auto run = [mode, tolerance, triangulation](const auto &forms) {
    auto [mesh, tag_labels, face_labels, curves] =
        tf::make_mesh_arrangements(
            forms,
            tf::arrangement_config{
                tf::intersect_config{static_cast<tf::intersect_mode>(mode),
                                     tolerance},
                static_cast<tf::triangulation_type>(triangulation)},
            tf::return_curves);
    auto mesh_pair = make_numpy_array(std::move(mesh));
    auto tag_array = make_numpy_array(std::move(tag_labels));
    auto face_array = make_numpy_array(std::move(face_labels));
    auto [paths, c_points] = make_numpy_array(std::move(curves));
    auto curve_tuple = nanobind::make_tuple(
        nanobind::make_tuple(paths.first, paths.second), std::move(c_points));
    return nanobind::make_tuple(std::move(mesh_pair), std::move(tag_array),
                                std::move(face_array), std::move(curve_tuple));
  };

  auto range = tf::make_range(wrappers.data(), wrappers.size());

  if (any_transformed) {
    auto identity = tf::make_identity_transformation<RealT, Dims>();
    auto tagged = tf::make_mapped_range(range, [&](auto &w) {
      auto fm = w.face_membership();
      auto mel = w.manifold_edge_link();
      tf::transformation<RealT, Dims> tv = w.has_transformation()
          ? tf::transformation<RealT, Dims>(w.transformation_view())
          : identity;
      return w.make_primitive_range() | tf::tag(w.tree()) | tf::tag(fm) |
             tf::tag(mel) | tf::tag(tv);
    });
    return run(tagged);
  }

  auto tagged = tf::make_mapped_range(range, [](auto &w) {
    auto fm = w.face_membership();
    auto mel = w.manifold_edge_link();
    return w.make_primitive_range() | tf::tag(w.tree()) | tf::tag(fm) |
           tf::tag(mel);
  });
  return run(tagged);
}

} // namespace tf::py
