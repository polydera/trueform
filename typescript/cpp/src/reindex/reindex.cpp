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

#include "trueform/reindex.hpp"
#include "trueform/ts/reindex/result_types.hpp"
#include "trueform/ts/core/promise.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

// ============================================================================
// Helper: reconstruct index_map view from wasm_index_map
// ============================================================================

auto to_index_map(wasm_index_map &im) {
  return tf::make_index_map(im.f.make_range(), im.kept_ids.make_range());
}

// ============================================================================
// reindexed(mesh, faceMap, pointMap) → Mesh
// ============================================================================

auto sync_reindexed_mesh(wasm_mesh &m, wasm_index_map &face_im,
                         wasm_index_map &point_im) -> wasm_mesh {
  auto poly = m.polygons_range();
  auto fim = to_index_map(face_im);
  auto pim = to_index_map(point_im);
  return wasm_mesh::from_polygons_buffer(tf::reindexed(poly, fim, pim));
}

auto async_reindexed_mesh(wasm_mesh &m, wasm_index_map &face_im,
                          wasm_index_map &point_im) -> promise_t {
  return promise([a = m, b = face_im, c = point_im]() -> wasm_mesh {
    return sync_reindexed_mesh(const_cast<wasm_mesh &>(a),
                               const_cast<wasm_index_map &>(b),
                               const_cast<wasm_index_map &>(c));
  });
}

// ============================================================================
// reindexed_by_mask(mesh, mask) ± maps
// ============================================================================

auto sync_reindexed_by_mask(wasm_mesh &m, wasm_ndarray<std::int8_t> &mask)
    -> wasm_mesh {
  auto poly = m.polygons_range();
  auto [result, face_im, point_im] =
      tf::reindexed_by_mask(poly, mask.make_range(), tf::return_index_map);
  return wasm_mesh::from_polygons_buffer(std::move(result));
}

auto sync_reindexed_by_mask_with_maps(wasm_mesh &m,
                                      wasm_ndarray<std::int8_t> &mask)
    -> reindexed_mesh_result {
  auto poly = m.polygons_range();
  auto [result, face_im, point_im] =
      tf::reindexed_by_mask(poly, mask.make_range(), tf::return_index_map);
  return {
      wasm_mesh::from_polygons_buffer(std::move(result)),
      wasm_index_map::from_index_map_buffer(std::move(face_im)),
      wasm_index_map::from_index_map_buffer(std::move(point_im)),
  };
}

auto async_reindexed_by_mask(wasm_mesh &m, wasm_ndarray<std::int8_t> &mask)
    -> promise_t {
  return promise([a = m, b = mask]() -> wasm_mesh {
    return sync_reindexed_by_mask(const_cast<wasm_mesh &>(a),
                                  const_cast<wasm_ndarray<std::int8_t> &>(b));
  });
}

auto async_reindexed_by_mask_with_maps(wasm_mesh &m,
                                       wasm_ndarray<std::int8_t> &mask)
    -> promise_t {
  return promise([a = m, b = mask]() -> reindexed_mesh_result {
    return sync_reindexed_by_mask_with_maps(
        const_cast<wasm_mesh &>(a),
        const_cast<wasm_ndarray<std::int8_t> &>(b));
  });
}

// ============================================================================
// reindexed_by_ids(mesh, ids) ± maps
// ============================================================================

auto sync_reindexed_by_ids(wasm_mesh &m, wasm_ndarray<int> &ids) -> wasm_mesh {
  auto poly = m.polygons_range();
  return wasm_mesh::from_polygons_buffer(
      tf::reindexed_by_ids(poly, ids.make_range()));
}

auto sync_reindexed_by_ids_with_maps(wasm_mesh &m, wasm_ndarray<int> &ids)
    -> reindexed_mesh_result {
  auto poly = m.polygons_range();
  auto [result, face_im, point_im] =
      tf::reindexed_by_ids(poly, ids.make_range(), tf::return_index_map);
  return {
      wasm_mesh::from_polygons_buffer(std::move(result)),
      wasm_index_map::from_index_map_buffer(std::move(face_im)),
      wasm_index_map::from_index_map_buffer(std::move(point_im)),
  };
}

auto async_reindexed_by_ids(wasm_mesh &m, wasm_ndarray<int> &ids)
    -> promise_t {
  return promise([a = m, b = ids]() -> wasm_mesh {
    return sync_reindexed_by_ids(const_cast<wasm_mesh &>(a),
                                 const_cast<wasm_ndarray<int> &>(b));
  });
}

auto async_reindexed_by_ids_with_maps(wasm_mesh &m, wasm_ndarray<int> &ids)
    -> promise_t {
  return promise([a = m, b = ids]() -> reindexed_mesh_result {
    return sync_reindexed_by_ids_with_maps(const_cast<wasm_mesh &>(a),
                                           const_cast<wasm_ndarray<int> &>(b));
  });
}

// ============================================================================
// reindexed_by_mask_on_points(mesh, pointMask) ± maps
// ============================================================================

auto sync_reindexed_by_mask_on_points(wasm_mesh &m,
                                      wasm_ndarray<std::int8_t> &mask)
    -> wasm_mesh {
  auto poly = m.polygons_range();
  auto [result, face_im, point_im] = tf::reindexed_by_mask_on_points(
      poly, mask.make_range(), tf::return_index_map);
  return wasm_mesh::from_polygons_buffer(std::move(result));
}

auto sync_reindexed_by_mask_on_points_with_maps(
    wasm_mesh &m, wasm_ndarray<std::int8_t> &mask) -> reindexed_mesh_result {
  auto poly = m.polygons_range();
  auto [result, face_im, point_im] = tf::reindexed_by_mask_on_points(
      poly, mask.make_range(), tf::return_index_map);
  return {
      wasm_mesh::from_polygons_buffer(std::move(result)),
      wasm_index_map::from_index_map_buffer(std::move(face_im)),
      wasm_index_map::from_index_map_buffer(std::move(point_im)),
  };
}

auto async_reindexed_by_mask_on_points(wasm_mesh &m,
                                       wasm_ndarray<std::int8_t> &mask)
    -> promise_t {
  return promise([a = m, b = mask]() -> wasm_mesh {
    return sync_reindexed_by_mask_on_points(
        const_cast<wasm_mesh &>(a),
        const_cast<wasm_ndarray<std::int8_t> &>(b));
  });
}

auto async_reindexed_by_mask_on_points_with_maps(
    wasm_mesh &m, wasm_ndarray<std::int8_t> &mask) -> promise_t {
  return promise([a = m, b = mask]() -> reindexed_mesh_result {
    return sync_reindexed_by_mask_on_points_with_maps(
        const_cast<wasm_mesh &>(a),
        const_cast<wasm_ndarray<std::int8_t> &>(b));
  });
}

// ============================================================================
// reindexed_by_ids_on_points(mesh, pointIds) ± maps
// ============================================================================

auto sync_reindexed_by_ids_on_points(wasm_mesh &m, wasm_ndarray<int> &ids)
    -> wasm_mesh {
  auto poly = m.polygons_range();
  auto [result, face_im, point_im] = tf::reindexed_by_ids_on_points(
      poly, ids.make_range(), tf::return_index_map);
  return wasm_mesh::from_polygons_buffer(std::move(result));
}

auto sync_reindexed_by_ids_on_points_with_maps(wasm_mesh &m,
                                                wasm_ndarray<int> &ids)
    -> reindexed_mesh_result {
  auto poly = m.polygons_range();
  auto [result, face_im, point_im] = tf::reindexed_by_ids_on_points(
      poly, ids.make_range(), tf::return_index_map);
  return {
      wasm_mesh::from_polygons_buffer(std::move(result)),
      wasm_index_map::from_index_map_buffer(std::move(face_im)),
      wasm_index_map::from_index_map_buffer(std::move(point_im)),
  };
}

auto async_reindexed_by_ids_on_points(wasm_mesh &m, wasm_ndarray<int> &ids)
    -> promise_t {
  return promise([a = m, b = ids]() -> wasm_mesh {
    return sync_reindexed_by_ids_on_points(const_cast<wasm_mesh &>(a),
                                           const_cast<wasm_ndarray<int> &>(b));
  });
}

auto async_reindexed_by_ids_on_points_with_maps(wasm_mesh &m,
                                                 wasm_ndarray<int> &ids)
    -> promise_t {
  return promise([a = m, b = ids]() -> reindexed_mesh_result {
    return sync_reindexed_by_ids_on_points_with_maps(
        const_cast<wasm_mesh &>(a), const_cast<wasm_ndarray<int> &>(b));
  });
}

// ============================================================================
// concatenate_meshes(js_array) → Mesh
// ============================================================================

auto sync_concatenate_meshes(emscripten::val js_meshes) -> wasm_mesh {
  auto n = js_meshes["length"].as<int>();
  if (n == 0)
    throw std::runtime_error("concatenateMeshes: empty array");

  std::vector<wasm_mesh> meshes;
  meshes.reserve(n);
  bool any_transformed = false;
  for (int i = 0; i < n; ++i) {
    meshes.push_back(
        js_meshes[i].as<wasm_mesh>(emscripten::allow_raw_pointers()));
    if (meshes.back().has_transformation())
      any_transformed = true;
  }

  auto mesh_range = tf::make_range(meshes);

  if (any_transformed) {
    auto identity = tf::make_identity_transformation<float, 3>();
    auto tagged = tf::make_mapped_range(mesh_range, [&](auto &m) {
      auto tv = m.has_transformation()
                    ? m.transformation_view()
                    : tf::make_transformation_view<3>(&identity(0, 0));
      return m.polygons_range() | tf::tag(tv);
    });
    return wasm_mesh::from_polygons_buffer(tf::concatenated(tagged));
  }

  auto plain = tf::make_mapped_range(mesh_range, [](auto &m) {
    return m.polygons_range();
  });
  return wasm_mesh::from_polygons_buffer(tf::concatenated(plain));
}

auto async_concatenate_meshes(emscripten::val js_meshes) -> promise_t {
  // Capture meshes by copying from JS array
  auto n = js_meshes["length"].as<int>();
  std::vector<wasm_mesh> meshes;
  meshes.reserve(n);
  for (int i = 0; i < n; ++i)
    meshes.push_back(
        js_meshes[i].as<wasm_mesh>(emscripten::allow_raw_pointers()));

  return promise([ms = std::move(meshes)]() -> wasm_mesh {
    auto &meshes = const_cast<std::vector<wasm_mesh> &>(ms);
    bool any_transformed = false;
    for (auto &m : meshes)
      if (m.has_transformation())
        any_transformed = true;

    auto mesh_range = tf::make_range(meshes);

    if (any_transformed) {
      auto identity = tf::make_identity_transformation<float, 3>();
      auto tagged = tf::make_mapped_range(mesh_range, [&](auto &m) {
        auto tv = m.has_transformation()
                      ? m.transformation_view()
                      : tf::make_transformation_view<3>(&identity(0, 0));
        return m.polygons_range() | tf::tag(tv);
      });
      return wasm_mesh::from_polygons_buffer(tf::concatenated(tagged));
    }

    auto plain = tf::make_mapped_range(mesh_range, [](auto &m) {
      return m.polygons_range();
    });
    return wasm_mesh::from_polygons_buffer(tf::concatenated(plain));
  });
}

// ============================================================================
// split_into_components(mesh, labels) → {components: Mesh[], labels: int[]}
// ============================================================================

auto sync_split_into_components(wasm_mesh &m, wasm_ndarray<int> &labels)
    -> emscripten::val {
  auto poly = m.polygons_range();
  auto [components, comp_labels] =
      tf::split_into_components(poly, labels.make_range());

  auto result = emscripten::val::object();
  auto js_meshes = emscripten::val::array();
  auto js_labels = emscripten::val::array();
  for (std::size_t i = 0; i < components.size(); ++i) {
    js_meshes.call<void>(
        "push", wasm_mesh::from_polygons_buffer(std::move(components[i])));
    js_labels.call<void>("push", comp_labels[i]);
  }
  result.set("components", js_meshes);
  result.set("labels", js_labels);
  return result;
}

auto async_split_into_components(wasm_mesh &m, wasm_ndarray<int> &labels)
    -> promise_t {
  return promise(
      [a = m, b = labels]() -> emscripten::val {
        return sync_split_into_components(const_cast<wasm_mesh &>(a),
                                          const_cast<wasm_ndarray<int> &>(b));
      });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_reindex) {
  // Result type
  emscripten::value_object<tf::ts::reindexed_mesh_result>(
      "ReindexedMeshResult")
      .field("mesh", &tf::ts::reindexed_mesh_result::mesh)
      .field("faceMap", &tf::ts::reindexed_mesh_result::face_map)
      .field("pointMap", &tf::ts::reindexed_mesh_result::point_map);

  // reindexed(mesh, faceMap, pointMap)
  emscripten::function("reindexed_mesh", &sync_reindexed_mesh);
  emscripten::function("dispatch_reindexed_mesh", &async_reindexed_mesh);

  // reindexed_by_mask
  emscripten::function("reindexed_by_mask", &sync_reindexed_by_mask);
  emscripten::function("reindexed_by_mask_with_maps",
                       &sync_reindexed_by_mask_with_maps);
  emscripten::function("dispatch_reindexed_by_mask",
                       &async_reindexed_by_mask);
  emscripten::function("dispatch_reindexed_by_mask_with_maps",
                       &async_reindexed_by_mask_with_maps);

  // reindexed_by_ids
  emscripten::function("reindexed_by_ids", &sync_reindexed_by_ids);
  emscripten::function("reindexed_by_ids_with_maps",
                       &sync_reindexed_by_ids_with_maps);
  emscripten::function("dispatch_reindexed_by_ids",
                       &async_reindexed_by_ids);
  emscripten::function("dispatch_reindexed_by_ids_with_maps",
                       &async_reindexed_by_ids_with_maps);

  // reindexed_by_mask_on_points
  emscripten::function("reindexed_by_mask_on_points",
                       &sync_reindexed_by_mask_on_points);
  emscripten::function("reindexed_by_mask_on_points_with_maps",
                       &sync_reindexed_by_mask_on_points_with_maps);
  emscripten::function("dispatch_reindexed_by_mask_on_points",
                       &async_reindexed_by_mask_on_points);
  emscripten::function("dispatch_reindexed_by_mask_on_points_with_maps",
                       &async_reindexed_by_mask_on_points_with_maps);

  // reindexed_by_ids_on_points
  emscripten::function("reindexed_by_ids_on_points",
                       &sync_reindexed_by_ids_on_points);
  emscripten::function("reindexed_by_ids_on_points_with_maps",
                       &sync_reindexed_by_ids_on_points_with_maps);
  emscripten::function("dispatch_reindexed_by_ids_on_points",
                       &async_reindexed_by_ids_on_points);
  emscripten::function("dispatch_reindexed_by_ids_on_points_with_maps",
                       &async_reindexed_by_ids_on_points_with_maps);

  // concatenate_meshes
  emscripten::function("concatenate_meshes", &sync_concatenate_meshes);
  emscripten::function("dispatch_concatenate_meshes",
                       &async_concatenate_meshes);

  // split_into_components
  emscripten::function("split_into_components", &sync_split_into_components);
  emscripten::function("dispatch_split_into_components",
                       &async_split_into_components);
}
