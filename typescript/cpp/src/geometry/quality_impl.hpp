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

#include "trueform/core/algorithm/parallel_transform.hpp"
#include "trueform/core/angle.hpp"
#include "trueform/core/buffer.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/policy/normals.hpp"
#include "trueform/core/unit_vectors.hpp"
#include "trueform/geometry/compute_dihedral_angles.hpp"
#include "trueform/geometry/compute_face_quality.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <utility>

namespace tf {
namespace ts {

// An angle crosses the layer as the number it is, so the strong type is
// stripped once, here, and the arrays the caller reads are radians.
template <typename Real>
auto radians_array(tf::buffer<tf::rad<Real>> &&angles, int count)
    -> wasm_ndarray<Real> {
  tf::buffer<Real> values;
  values.allocate(angles.size());
  tf::parallel_transform(
      angles, values, [](tf::rad<Real> angle) { return angle.value; },
      tf::checked);
  return wasm_ndarray<Real>::from_buffer(std::move(values), {count});
}

template <typename Real> struct face_quality_result_t {
  wasm_ndarray<Real> quality;      // [F]
  wasm_ndarray<Real> min_angle;    // [F]
  wasm_ndarray<Real> max_angle;    // [F]
  wasm_ndarray<Real> aspect_ratio; // [F]
};

template <typename Real> struct dihedral_angles_result_t {
  wasm_ndarray<int> edges;   // [N, 2]
  wasm_ndarray<Real> angles; // [N]
};

template <typename Real>
auto sync_face_quality(wasm_mesh<Real> &m) -> face_quality_result_t<Real> {
  auto measured = tf::compute_face_quality(m.polygons_range());
  const auto count = m.number_of_faces();
  return {wasm_ndarray<Real>::from_buffer(std::move(measured.quality), {count}),
          radians_array<Real>(std::move(measured.min_angle), count),
          radians_array<Real>(std::move(measured.max_angle), count),
          wasm_ndarray<Real>::from_buffer(std::move(measured.aspect_ratio),
                                          {count})};
}

template <typename Real>
auto sync_dihedral_angles(wasm_mesh<Real> &m)
    -> dihedral_angles_result_t<Real> {
  auto mel = m.manifold_edge_link_range();
  auto n_arr = m.normals();
  auto n_view = tf::make_unit_vectors<3>(n_arr.make_range());
  auto measured = tf::compute_dihedral_angles(
      m.polygons_range() | tf::tag(mel) | tf::tag_normals(n_view));
  const auto count = static_cast<int>(measured.edges.size());
  return {wasm_ndarray<int>::from_buffer(std::move(measured.edges.data_buffer()),
                                         {count, 2}),
          radians_array<Real>(std::move(measured.angles), count)};
}

template <typename Real>
auto async_face_quality(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> face_quality_result_t<Real> {
    return sync_face_quality<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

template <typename Real>
auto async_dihedral_angles(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> dihedral_angles_result_t<Real> {
    return sync_dihedral_angles<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

} // namespace ts
} // namespace tf
