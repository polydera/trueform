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

#include "trueform/intersect/intersect_mode.hpp"
#include "trueform/intersect/make_intersection_curves.hpp"
#include "trueform/intersect/make_isocurves.hpp"
#include "trueform/intersect/make_self_intersection_curves.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_curves.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>

namespace {

using namespace tf::ts;

// ============================================================================
// Intersection curves (mesh0 x mesh1 -> curves)
// ============================================================================

auto sync_intersection_curves(wasm_mesh &m0, wasm_mesh &m1, int mode)
    -> wasm_curves {
  bool has0 = m0.has_transformation();
  bool has1 = m1.has_transformation();
  auto fm0 = m0.face_membership_range();
  auto mel0 = m0.manifold_edge_link_range();
  auto fm1 = m1.face_membership_range();
  auto mel1 = m1.manifold_edge_link_range();
  auto make_return = [&](auto &&poly0, auto &&poly1) -> wasm_curves {
    auto cb = tf::make_intersection_curves(
        poly0 | tf::tag(m0.tree()) | tf::tag(fm0) | tf::tag(mel0),
        poly1 | tf::tag(m1.tree()) | tf::tag(fm1) | tf::tag(mel1),
        static_cast<tf::intersect_mode>(mode));
    return wasm_curves::from_curves_buffer(std::move(cb));
  };
  if (has0 && has1)
    return make_return(
        m0.polygons_range() | tf::tag(m0.transformation_view()),
        m1.polygons_range() | tf::tag(m1.transformation_view()));
  else if (has0)
    return make_return(
        m0.polygons_range() | tf::tag(m0.transformation_view()),
        m1.polygons_range());
  else if (has1)
    return make_return(m0.polygons_range(),
                       m1.polygons_range() | tf::tag(m1.transformation_view()));
  else
    return make_return(m0.polygons_range(), m1.polygons_range());
}

auto async_intersection_curves(wasm_mesh &m0, wasm_mesh &m1, int mode)
    -> promise_t {
  return promise([a = m0, b = m1, mode]() -> wasm_curves {
    return sync_intersection_curves(const_cast<wasm_mesh &>(a),
                                    const_cast<wasm_mesh &>(b), mode);
  });
}

// ============================================================================
// Intersection curves list (N meshes x mode -> curves)
// ============================================================================

auto extract_meshes(emscripten::val js_meshes) -> std::vector<wasm_mesh> {
  auto n = js_meshes["length"].as<int>();
  std::vector<wasm_mesh> meshes;
  meshes.reserve(n);
  for (int i = 0; i < n; ++i)
    meshes.push_back(
        js_meshes[i].as<wasm_mesh>(emscripten::allow_raw_pointers()));
  return meshes;
}

auto intersection_curves_list_impl(std::vector<wasm_mesh> &meshes, int mode)
    -> wasm_curves {
  auto mesh_range = tf::make_range(meshes);
  bool any_transformed = false;
  for (auto &m : meshes)
    if (m.has_transformation())
      any_transformed = true;

  auto run = [mode](const auto &forms) -> wasm_curves {
    auto cb = tf::make_intersection_curves(
        forms, static_cast<tf::intersect_mode>(mode));
    return wasm_curves::from_curves_buffer(std::move(cb));
  };

  if (any_transformed) {
    auto identity = tf::make_identity_transformation<float, 3>();
    auto tagged = tf::make_mapped_range(mesh_range, [&](auto &m) {
      auto fm = m.face_membership_range();
      auto mel = m.manifold_edge_link_range();
      auto tv = m.has_transformation()
                    ? m.transformation_view()
                    : tf::make_transformation_view<3>(&identity(0, 0));
      return m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) |
             tf::tag(mel) | tf::tag(tv);
    });
    return run(tagged);
  }

  auto tagged = tf::make_mapped_range(mesh_range, [](auto &m) {
    auto fm = m.face_membership_range();
    auto mel = m.manifold_edge_link_range();
    return m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) |
           tf::tag(mel);
  });
  return run(tagged);
}

auto sync_intersection_curves_list(emscripten::val js_meshes, int mode)
    -> wasm_curves {
  auto meshes = extract_meshes(js_meshes);
  return intersection_curves_list_impl(meshes, mode);
}

auto async_intersection_curves_list(emscripten::val js_meshes, int mode)
    -> promise_t {
  auto meshes = extract_meshes(js_meshes);
  return promise([ms = std::move(meshes), mode]() -> wasm_curves {
    auto &meshes = const_cast<std::vector<wasm_mesh> &>(ms);
    return intersection_curves_list_impl(meshes, mode);
  });
}

// ============================================================================
// Self-intersection curves (mesh -> curves)
// ============================================================================

auto sync_self_intersection_curves(wasm_mesh &m, int mode) -> wasm_curves {
  auto fm = m.face_membership_range();
  auto mel = m.manifold_edge_link_range();
  auto cb = tf::make_self_intersection_curves(
      m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) | tf::tag(mel),
      static_cast<tf::intersect_mode>(mode));
  return wasm_curves::from_curves_buffer(std::move(cb));
}

auto async_self_intersection_curves(wasm_mesh &m, int mode) -> promise_t {
  return promise([a = m, mode]() -> wasm_curves {
    return sync_self_intersection_curves(const_cast<wasm_mesh &>(a), mode);
  });
}

// ============================================================================
// Isocontours — single threshold
// ============================================================================

auto sync_isocontours(wasm_mesh &mesh, wasm_ndarray<float> &scalars,
                      float cut_value) -> wasm_curves {
  auto cb = tf::make_isocontours(mesh.polygons_range(), scalars.make_range(),
                                 cut_value);
  return wasm_curves::from_curves_buffer(std::move(cb));
}

auto async_isocontours(wasm_mesh &mesh, wasm_ndarray<float> &scalars,
                       float cut_value) -> promise_t {
  return promise([m = mesh, s = scalars, cut_value]() -> wasm_curves {
    return sync_isocontours(const_cast<wasm_mesh &>(m),
                            const_cast<wasm_ndarray<float> &>(s), cut_value);
  });
}

// ============================================================================
// Isocontours — multiple thresholds
// ============================================================================

auto extract_cut_values(emscripten::val js_cut_values) -> std::vector<float> {
  auto n = js_cut_values["length"].as<std::size_t>();
  std::vector<float> cv(n);
  for (std::size_t i = 0; i < n; ++i)
    cv[i] = js_cut_values[i].as<float>();
  return cv;
}

auto sync_isocontours_multi(wasm_mesh &mesh, wasm_ndarray<float> &scalars,
                            emscripten::val js_cut_values) -> wasm_curves {
  auto cv = extract_cut_values(js_cut_values);
  auto cb = tf::make_isocontours(mesh.polygons_range(), scalars.make_range(),
                                 tf::make_range(cv.data(), cv.size()));
  return wasm_curves::from_curves_buffer(std::move(cb));
}

auto async_isocontours_multi(wasm_mesh &mesh, wasm_ndarray<float> &scalars,
                             emscripten::val js_cut_values) -> promise_t {
  auto cv = extract_cut_values(js_cut_values);
  return promise(
      [m = mesh, s = scalars, cv = std::move(cv)]() -> wasm_curves {
        auto cb = tf::make_isocontours(
            const_cast<wasm_mesh &>(m).polygons_range(),
            const_cast<wasm_ndarray<float> &>(s).make_range(),
            tf::make_range(cv.data(), cv.size()));
        return wasm_curves::from_curves_buffer(std::move(cb));
      });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_intersect) {
  // Intersection curves — sync/async
  emscripten::function("intersection_curves", &sync_intersection_curves);
  emscripten::function("dispatch_intersection_curves",
                       &async_intersection_curves);

  // Self-intersection curves — sync/async
  emscripten::function("self_intersection_curves",
                       &sync_self_intersection_curves);
  emscripten::function("dispatch_self_intersection_curves",
                       &async_self_intersection_curves);

  // Isocontours (single threshold) — sync/async
  emscripten::function("isocontours", &sync_isocontours);
  emscripten::function("dispatch_isocontours", &async_isocontours);

  // Intersection curves list — sync/async
  emscripten::function("intersection_curves_list",
                       &sync_intersection_curves_list);
  emscripten::function("dispatch_intersection_curves_list",
                       &async_intersection_curves_list);

  // Isocontours (multiple thresholds) — sync/async
  emscripten::function("isocontours_multi", &sync_isocontours_multi);
  emscripten::function("dispatch_isocontours_multi", &async_isocontours_multi);
}
