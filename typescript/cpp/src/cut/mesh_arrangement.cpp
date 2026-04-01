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

#include "trueform/cut/make_mesh_arrangements.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/cut/result_types.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

struct arrangement_result {
  wasm_mesh mesh;
  wasm_ndarray<int> tag_labels;
  wasm_ndarray<int> face_labels;
};

struct arrangement_result_with_curves {
  wasm_mesh mesh;
  wasm_ndarray<int> tag_labels;
  wasm_ndarray<int> face_labels;
  wasm_curves curves;
};

auto extract_meshes(emscripten::val js_meshes) -> std::vector<wasm_mesh> {
  auto n = js_meshes["length"].as<int>();
  if (n < 2)
    throw std::runtime_error("meshArrangement: need at least 2 meshes");
  std::vector<wasm_mesh> meshes;
  meshes.reserve(n);
  for (int i = 0; i < n; ++i)
    meshes.push_back(
        js_meshes[i].as<wasm_mesh>(emscripten::allow_raw_pointers()));
  return meshes;
}

template <typename MeshRange>
auto run_arrangement(MeshRange &meshes, int mode) -> arrangement_result {
  auto mesh_range = tf::make_range(meshes);
  bool any_transformed = false;
  for (auto &m : meshes)
    if (m.has_transformation())
      any_transformed = true;

  auto run = [mode](const auto &forms) -> arrangement_result {
    auto [mesh, tag_labels, face_labels] = tf::make_mesh_arrangements(
        forms, static_cast<tf::intersect_mode>(mode));
    return {wasm_mesh::from_polygons_buffer(std::move(mesh)),
            wasm_ndarray<int>::from_buffer(std::move(tag_labels)),
            wasm_ndarray<int>::from_buffer(std::move(face_labels))};
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

template <typename MeshRange>
auto run_arrangement_with_curves(MeshRange &meshes, int mode)
    -> arrangement_result_with_curves {
  auto mesh_range = tf::make_range(meshes);
  bool any_transformed = false;
  for (auto &m : meshes)
    if (m.has_transformation())
      any_transformed = true;

  auto run = [mode](const auto &forms) -> arrangement_result_with_curves {
    auto [mesh, tag_labels, face_labels, curves] =
        tf::make_mesh_arrangements(
            forms, static_cast<tf::intersect_mode>(mode), tf::return_curves);
    return {wasm_mesh::from_polygons_buffer(std::move(mesh)),
            wasm_ndarray<int>::from_buffer(std::move(tag_labels)),
            wasm_ndarray<int>::from_buffer(std::move(face_labels)),
            wasm_curves::from_curves_buffer(std::move(curves))};
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

// -- Sync --

auto sync_mesh_arrangement(emscripten::val js_meshes, int mode)
    -> arrangement_result {
  auto meshes = extract_meshes(js_meshes);
  return run_arrangement(meshes, mode);
}

auto sync_mesh_arrangement_with_curves(emscripten::val js_meshes, int mode)
    -> arrangement_result_with_curves {
  auto meshes = extract_meshes(js_meshes);
  return run_arrangement_with_curves(meshes, mode);
}

// -- Async --

auto async_mesh_arrangement(emscripten::val js_meshes, int mode) -> promise_t {
  auto meshes = extract_meshes(js_meshes);
  return promise([ms = std::move(meshes), mode]() -> arrangement_result {
    auto &meshes = const_cast<std::vector<wasm_mesh> &>(ms);
    return run_arrangement(meshes, mode);
  });
}

auto async_mesh_arrangement_with_curves(emscripten::val js_meshes, int mode)
    -> promise_t {
  auto meshes = extract_meshes(js_meshes);
  return promise(
      [ms = std::move(meshes), mode]() -> arrangement_result_with_curves {
        auto &meshes = const_cast<std::vector<wasm_mesh> &>(ms);
        return run_arrangement_with_curves(meshes, mode);
      });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_mesh_arrangement) {
  emscripten::value_object<arrangement_result>("MeshArrangementResult")
      .field("mesh", &arrangement_result::mesh)
      .field("tagLabels", &arrangement_result::tag_labels)
      .field("faceLabels", &arrangement_result::face_labels);

  emscripten::value_object<arrangement_result_with_curves>(
      "MeshArrangementResultWithCurves")
      .field("mesh", &arrangement_result_with_curves::mesh)
      .field("tagLabels", &arrangement_result_with_curves::tag_labels)
      .field("faceLabels", &arrangement_result_with_curves::face_labels)
      .field("curves", &arrangement_result_with_curves::curves);

  emscripten::function("mesh_arrangements", &sync_mesh_arrangement);
  emscripten::function("mesh_arrangements_with_curves",
                       &sync_mesh_arrangement_with_curves);
  emscripten::function("dispatch_mesh_arrangements", &async_mesh_arrangement);
  emscripten::function("dispatch_mesh_arrangements_with_curves",
                       &async_mesh_arrangement_with_curves);
}
