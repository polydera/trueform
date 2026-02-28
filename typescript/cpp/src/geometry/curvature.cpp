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

#include "trueform/geometry/compute_principal_curvatures.hpp"
#include "trueform/geometry/compute_shape_index.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

struct curvature_result {
  wasm_ndarray<float> k0;
  wasm_ndarray<float> k1;
};

struct curvature_directions_result {
  wasm_ndarray<float> k0;
  wasm_ndarray<float> k1;
  wasm_ndarray<float> d0;
  wasm_ndarray<float> d1;
};

// Tag polygons with cached fm + vertex_link + point normals
template <typename Fn>
auto with_tagged_polys(wasm_mesh &m, Fn &&fn) -> decltype(auto) {
  auto fm = m.face_membership_range();
  auto vl = m.vertex_link_range();

  // Ensure point normals are cached, create unit_vectors view
  auto pn_arr = m.point_normals();
  auto pn_view = tf::make_unit_vectors<3>(pn_arr.make_range());

  // Tag points with normals, reconstruct polygons with tagged points
  auto polys = m.polygons_range();
  auto tagged_points = polys.points() | tf::tag_normals(pn_view);
  auto tagged =
      tf::make_polygons(polys.faces(), tagged_points) | tf::tag(fm) | tf::tag(vl);

  return fn(tagged);
}

// -- Sync --

auto sync_principal_curvatures(wasm_mesh &m, int k) -> curvature_result {
  return with_tagged_polys(m, [&](auto &&polys) {
    auto [k0, k1] =
        tf::make_principal_curvatures(polys, static_cast<std::size_t>(k));
    const auto nv = m.number_of_points();
    return curvature_result{
        wasm_ndarray<float>::from_buffer(std::move(k0), {nv}),
        wasm_ndarray<float>::from_buffer(std::move(k1), {nv}),
    };
  });
}

auto sync_principal_directions(wasm_mesh &m, int k)
    -> curvature_directions_result {
  return with_tagged_polys(m, [&](auto &&polys) {
    auto [k0, k1, d0, d1] =
        tf::make_principal_directions(polys, static_cast<std::size_t>(k));
    const auto nv = m.number_of_points();
    return curvature_directions_result{
        wasm_ndarray<float>::from_buffer(std::move(k0), {nv}),
        wasm_ndarray<float>::from_buffer(std::move(k1), {nv}),
        wasm_ndarray<float>::from_buffer(std::move(d0.data_buffer()), {nv, 3}),
        wasm_ndarray<float>::from_buffer(std::move(d1.data_buffer()), {nv, 3}),
    };
  });
}

auto sync_shape_index(wasm_mesh &m, int k) -> wasm_ndarray<float> {
  return with_tagged_polys(m, [&](auto &&polys) {
    auto si = tf::make_shape_index(polys, static_cast<std::size_t>(k));
    return wasm_ndarray<float>::from_buffer(std::move(si),
                                            {m.number_of_points()});
  });
}

// -- Async --

auto async_principal_curvatures(wasm_mesh &m, int k) -> promise_t {
  return promise([a = m, k]() -> curvature_result {
    return sync_principal_curvatures(const_cast<wasm_mesh &>(a), k);
  });
}

auto async_principal_directions(wasm_mesh &m, int k) -> promise_t {
  return promise([a = m, k]() -> curvature_directions_result {
    return sync_principal_directions(const_cast<wasm_mesh &>(a), k);
  });
}

auto async_shape_index(wasm_mesh &m, int k) -> promise_t {
  return promise([a = m, k]() -> wasm_ndarray<float> {
    return sync_shape_index(const_cast<wasm_mesh &>(a), k);
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_geometry_curvature) {
  emscripten::value_object<curvature_result>("CurvatureResult")
      .field("k0", &curvature_result::k0)
      .field("k1", &curvature_result::k1);

  emscripten::value_object<curvature_directions_result>(
      "CurvatureDirectionsResult")
      .field("k0", &curvature_directions_result::k0)
      .field("k1", &curvature_directions_result::k1)
      .field("d0", &curvature_directions_result::d0)
      .field("d1", &curvature_directions_result::d1);

  // Sync
  emscripten::function("principal_curvatures", &sync_principal_curvatures);
  emscripten::function("principal_directions", &sync_principal_directions);
  emscripten::function("shape_index", &sync_shape_index);

  // Async
  emscripten::function("dispatch_principal_curvatures",
                       &async_principal_curvatures);
  emscripten::function("dispatch_principal_directions",
                       &async_principal_directions);
  emscripten::function("dispatch_shape_index", &async_shape_index);
}
