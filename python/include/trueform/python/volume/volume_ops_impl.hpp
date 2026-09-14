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

#include "../util/make_numpy_array.hpp"
#include "./to_python_volume.hpp"
#include "./volume_real_request.hpp"
#include "./volume_wrapper.hpp"
#include <array>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <string>
#include <trueform/core/frame.hpp>
#include <trueform/core/policy/frame.hpp>
#include <trueform/core/range.hpp>
#include <trueform/volume/boolean_op.hpp>
#include <trueform/volume/isosurface_config.hpp>
#include <trueform/volume/make_boolean.hpp>
#include <trueform/volume/make_isocontours.hpp>
#include <trueform/volume/make_isosurface.hpp>
#include <trueform/volume/make_resampled_volume.hpp>
#include <trueform/volume/make_sphere_sdf.hpp>
#include <tuple>
#include <utility>

namespace tf::py {

inline auto volume_boolean_op_from_int(int op) -> tf::volume_boolean_op {
  switch (op) {
  case 1:
    return tf::volume_boolean_op::intersection;
  case 2:
    return tf::volume_boolean_op::difference;
  default:
    return tf::volume_boolean_op::union_;
  }
}

template <typename T, typename Coord>
auto isosurface_impl(volume_wrapper<T, Coord> &vol, double iso, int method,
                     bool refine, double stabilizer, int dtype) {
  tf::isosurface_config config;
  config.method = method == 1 ? tf::isosurface_method::dual_contouring
                              : tf::isosurface_method::flying_edges;
  config.refine = refine;
  config.stabilizer = stabilizer;
  return with_requested_real(dtype, [&](auto real) {
    using Real = decltype(real);
    const auto run = [&](const auto &v) {
      auto mesh = tf::make_isosurface<int, Real>(v, static_cast<Real>(iso),
                                                 config);
      auto [faces, points] = make_numpy_array(std::move(mesh));
      return nanobind::make_tuple(faces, points);
    };
    if (vol.has_transformation())
      return run(vol.view() |
                 tf::tag(tf::make_frame(vol.transformation_view())));
    return run(vol.view());
  });
}

template <typename RealT>
auto make_sphere_sdf_impl(std::array<int, 3> dims,
                          std::array<double, 3> spacing,
                          std::array<double, 3> origin,
                          std::array<double, 3> center, double radius) {
  return to_python_volume(tf::make_sphere_sdf<RealT>(
      dims, to_volume_point<RealT>(spacing), to_volume_point<RealT>(origin),
      to_volume_point<RealT>(center), static_cast<RealT>(radius)));
}

template <typename T, typename Coord>
auto resampled_volume_impl(volume_wrapper<T, Coord> &vol,
                           std::array<int, 3> dims,
                           std::array<double, 3> spacing,
                           std::array<double, 3> origin, int dtype) {
  return with_requested_real(dtype, [&](auto real) {
    using Real = decltype(real);
    const auto run = [&](const auto &v) {
      return to_python_volume(tf::make_resampled_volume<Real>(
          v, dims, to_volume_point<Real>(spacing),
          to_volume_point<Real>(origin)));
    };
    if (vol.has_transformation())
      return run(vol.view() |
                 tf::tag(tf::make_frame(vol.transformation_view())));
    return run(vol.view());
  });
}

template <typename T, typename Coord>
auto volume_boolean_impl(volume_wrapper<T, Coord> &a,
                         volume_wrapper<T, Coord> &b, int op, int dtype) {
  return with_requested_real(dtype, [&](auto real) {
    using Real = decltype(real);
    // A posed operand makes the pose part of the answer — the producer's own
    // fact, read off the views this call built.
    const auto run = [&](const auto &va, const auto &vb) -> nanobind::tuple {
      auto result =
          tf::make_boolean<Real>(va, vb, volume_boolean_op_from_int(op));
      if constexpr (tf::has_frame_policy<decltype(va)> ||
                    tf::has_frame_policy<decltype(vb)>) {
        auto pose = make_numpy_array(std::get<1>(result));
        auto field = to_python_volume(std::move(std::get<0>(result)));
        return nanobind::make_tuple(field[0], field[1], field[2], field[3],
                                    pose);
      } else {
        auto field = to_python_volume(std::move(result));
        return nanobind::make_tuple(field[0], field[1], field[2], field[3],
                                    nanobind::none());
      }
    };
    const bool pa = a.has_transformation();
    const bool pb = b.has_transformation();
    if (pa && pb)
      return run(a.view() | tf::tag(tf::make_frame(a.transformation_view())),
                 b.view() | tf::tag(tf::make_frame(b.transformation_view())));
    if (pa)
      return run(a.view() | tf::tag(tf::make_frame(a.transformation_view())),
                 b.view());
    if (pb)
      return run(a.view(),
                 b.view() | tf::tag(tf::make_frame(b.transformation_view())));
    return run(a.view(), b.view());
  });
}

/// The isovalues are field values in the type the CALL decides in, not in the
/// sample storage type, so they cross as doubles for every row and
/// @ref tf::make_isocontours casts each one once.
template <typename T, typename Coord>
auto volume_slice_contours_impl(
    volume_wrapper<T, Coord> &vol, std::array<double, 3> plane_origin,
    std::array<double, 3> u, std::array<double, 3> v, std::array<int, 2> dims2,
    std::array<double, 2> spacing2,
    nanobind::ndarray<nanobind::numpy, const double, nanobind::ndim<1>,
                      nanobind::c_contig>
        isovalues,
    int dtype) {
  return with_requested_real(dtype, [&](auto real) {
    using Real = decltype(real);
    const auto run = [&](const auto &v_) {
      auto curves = tf::make_isocontours<int, Real>(
          v_, to_volume_point<Real>(plane_origin), to_volume_point<Real>(u),
          to_volume_point<Real>(v), dims2,
          std::array<Real, 2>{static_cast<Real>(spacing2[0]),
                              static_cast<Real>(spacing2[1])},
          tf::make_range(isovalues.data(),
                         isovalues.data() + isovalues.size()));
      auto [paths, points] = make_numpy_array(std::move(curves));
      return nanobind::make_tuple(
          nanobind::make_tuple(paths.first, paths.second), std::move(points));
    };
    if (vol.has_transformation())
      return run(vol.view() |
                 tf::tag(tf::make_frame(vol.transformation_view())));
    return run(vol.view());
  });
}

/// One sample row: its wrapper plus the entries that CONSUME a field of that
/// storage type. Every sample type the layer accepts carries these.
template <typename T, typename Coord>
auto register_volume_ops(nanobind::module_ &m, const char *wrapper_name,
                         const char *suffix) -> void {
  register_volume_wrapper<T, Coord>(m, wrapper_name);

  std::string sfx = suffix;

  m.def(("isosurface_" + sfx).c_str(), &isosurface_impl<T, Coord>,
        nanobind::arg("volume"), nanobind::arg("iso"), nanobind::arg("method"),
        nanobind::arg("refine"), nanobind::arg("stabilizer"),
        nanobind::arg("dtype"),
        "Extract the isosurface of a scalar volume as a triangle mesh.");


  m.def(("volume_slice_contours_" + sfx).c_str(),
        &volume_slice_contours_impl<T, Coord>, nanobind::arg("volume"),
        nanobind::arg("plane_origin"), nanobind::arg("u"), nanobind::arg("v"),
        nanobind::arg("dims2"), nanobind::arg("spacing2"),
        nanobind::arg("isovalues"), nanobind::arg("dtype"),
        "Isocontours of a volume on an oriented slice plane, as 3D curves.");
}

/// The boolean of a row whose samples have a negative side; an unsigned field
/// is not an SDF, so those rows are never registered and the facade refuses
/// them by dtype.
template <typename T, typename Coord>
auto register_volume_boolean(nanobind::module_ &m, const char *suffix) -> void {
  m.def((std::string("volume_boolean_") + suffix).c_str(),
        &volume_boolean_impl<T, Coord>, nanobind::arg("a"), nanobind::arg("b"),
        nanobind::arg("op"), nanobind::arg("dtype"),
        "Boolean CSG of two signed distance fields, returned as a new field.");
}

/// The regrid of a row the facade exposes today: the floating rows. An
/// integer-sourced regrid arrives with its own rows once the facade states
/// them; the row table refuses the rest by dtype.
template <typename T, typename Coord>
auto register_volume_resample(nanobind::module_ &m, const char *suffix)
    -> void {
  m.def((std::string("resampled_volume_") + suffix).c_str(),
        &resampled_volume_impl<T, Coord>, nanobind::arg("volume"),
        nanobind::arg("dims"), nanobind::arg("spacing"),
        nanobind::arg("origin"), nanobind::arg("dtype"),
        "Resample a scalar field onto a stated grid, as a new field.");
}

/// A field GENERATOR is keyed by the type it emits in, not by any input's
/// storage, so its rows are the real ones alone.
template <typename RealT>
auto register_volume_generators(nanobind::module_ &m, const char *suffix)
    -> void {
  m.def((std::string("make_sphere_sdf_") + suffix).c_str(),
        &make_sphere_sdf_impl<RealT>, nanobind::arg("dims"),
        nanobind::arg("spacing"), nanobind::arg("origin"),
        nanobind::arg("center"), nanobind::arg("radius"),
        "Sample the signed distance field of a sphere onto a regular grid.");
}

} // namespace tf::py
