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

#include "trueform/reindex.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_index_map.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <cstdint>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tf {
namespace ts {

// ============================================================================
// Result types — reindex outputs carry the mesh's coordinate dtype; index
// maps and per-component label arrays stay int32.
// ============================================================================

template <typename Real> struct reindexed_mesh_result_t {
  wasm_mesh<Real> mesh;
  wasm_index_map face_map;
  wasm_index_map point_map;
};

template <typename Real> struct split_components_result_t {
  std::vector<wasm_mesh<Real>> components;
  wasm_ndarray<std::int32_t> labels;
};

// ============================================================================
// Helper — reconstruct an index_map view from a wasm_index_map handle.
// ============================================================================

inline auto to_index_map(wasm_index_map &im) {
  return tf::make_index_map(im.f.make_range(), im.kept_ids.make_range());
}

// ============================================================================
// reindexed(mesh, faceMap, pointMap) → Mesh
// ============================================================================

template <typename Real>
auto sync_reindexed_mesh(wasm_mesh<Real> &m, wasm_index_map &face_im,
                         wasm_index_map &point_im) -> wasm_mesh<Real> {
  auto poly = m.polygons_range();
  auto fim = to_index_map(face_im);
  auto pim = to_index_map(point_im);
  return wasm_mesh<Real>::from_polygons_buffer(tf::reindexed(poly, fim, pim));
}

template <typename Real>
auto async_reindexed_mesh(wasm_mesh<Real> &m, wasm_index_map &face_im,
                          wasm_index_map &point_im) -> promise_t {
  return promise([a = m, b = face_im, c = point_im]() -> wasm_mesh<Real> {
    return sync_reindexed_mesh<Real>(const_cast<wasm_mesh<Real> &>(a),
                                     const_cast<wasm_index_map &>(b),
                                     const_cast<wasm_index_map &>(c));
  });
}

// ============================================================================
// reindexed_by_mask(mesh, mask) ± maps
// ============================================================================

template <typename Real>
auto sync_reindexed_by_mask(wasm_mesh<Real> &m,
                            wasm_ndarray<std::int8_t> &mask)
    -> wasm_mesh<Real> {
  auto poly = m.polygons_range();
  auto [result, face_im, point_im] =
      tf::reindexed_by_mask(poly, mask.make_range(), tf::return_index_map);
  return wasm_mesh<Real>::from_polygons_buffer(std::move(result));
}

template <typename Real>
auto sync_reindexed_by_mask_with_maps(wasm_mesh<Real> &m,
                                      wasm_ndarray<std::int8_t> &mask)
    -> reindexed_mesh_result_t<Real> {
  auto poly = m.polygons_range();
  auto [result, face_im, point_im] =
      tf::reindexed_by_mask(poly, mask.make_range(), tf::return_index_map);
  return {
      wasm_mesh<Real>::from_polygons_buffer(std::move(result)),
      wasm_index_map::from_index_map_buffer(std::move(face_im)),
      wasm_index_map::from_index_map_buffer(std::move(point_im)),
  };
}

template <typename Real>
auto async_reindexed_by_mask(wasm_mesh<Real> &m,
                             wasm_ndarray<std::int8_t> &mask) -> promise_t {
  return promise([a = m, b = mask]() -> wasm_mesh<Real> {
    return sync_reindexed_by_mask<Real>(
        const_cast<wasm_mesh<Real> &>(a),
        const_cast<wasm_ndarray<std::int8_t> &>(b));
  });
}

template <typename Real>
auto async_reindexed_by_mask_with_maps(wasm_mesh<Real> &m,
                                       wasm_ndarray<std::int8_t> &mask)
    -> promise_t {
  return promise([a = m, b = mask]() -> reindexed_mesh_result_t<Real> {
    return sync_reindexed_by_mask_with_maps<Real>(
        const_cast<wasm_mesh<Real> &>(a),
        const_cast<wasm_ndarray<std::int8_t> &>(b));
  });
}

// ============================================================================
// reindexed_by_ids(mesh, ids) ± maps
// ============================================================================

template <typename Real>
auto sync_reindexed_by_ids(wasm_mesh<Real> &m, wasm_ndarray<int> &ids)
    -> wasm_mesh<Real> {
  auto poly = m.polygons_range();
  return wasm_mesh<Real>::from_polygons_buffer(
      tf::reindexed_by_ids(poly, ids.make_range()));
}

template <typename Real>
auto sync_reindexed_by_ids_with_maps(wasm_mesh<Real> &m, wasm_ndarray<int> &ids)
    -> reindexed_mesh_result_t<Real> {
  auto poly = m.polygons_range();
  auto [result, face_im, point_im] =
      tf::reindexed_by_ids(poly, ids.make_range(), tf::return_index_map);
  return {
      wasm_mesh<Real>::from_polygons_buffer(std::move(result)),
      wasm_index_map::from_index_map_buffer(std::move(face_im)),
      wasm_index_map::from_index_map_buffer(std::move(point_im)),
  };
}

template <typename Real>
auto async_reindexed_by_ids(wasm_mesh<Real> &m, wasm_ndarray<int> &ids)
    -> promise_t {
  return promise([a = m, b = ids]() -> wasm_mesh<Real> {
    return sync_reindexed_by_ids<Real>(const_cast<wasm_mesh<Real> &>(a),
                                       const_cast<wasm_ndarray<int> &>(b));
  });
}

template <typename Real>
auto async_reindexed_by_ids_with_maps(wasm_mesh<Real> &m,
                                      wasm_ndarray<int> &ids) -> promise_t {
  return promise([a = m, b = ids]() -> reindexed_mesh_result_t<Real> {
    return sync_reindexed_by_ids_with_maps<Real>(
        const_cast<wasm_mesh<Real> &>(a), const_cast<wasm_ndarray<int> &>(b));
  });
}

// ============================================================================
// reindexed_by_mask_on_points(mesh, pointMask) ± maps
// ============================================================================

template <typename Real>
auto sync_reindexed_by_mask_on_points(wasm_mesh<Real> &m,
                                      wasm_ndarray<std::int8_t> &mask)
    -> wasm_mesh<Real> {
  auto poly = m.polygons_range();
  auto [result, face_im, point_im] = tf::reindexed_by_mask_on_points(
      poly, mask.make_range(), tf::return_index_map);
  return wasm_mesh<Real>::from_polygons_buffer(std::move(result));
}

template <typename Real>
auto sync_reindexed_by_mask_on_points_with_maps(
    wasm_mesh<Real> &m, wasm_ndarray<std::int8_t> &mask)
    -> reindexed_mesh_result_t<Real> {
  auto poly = m.polygons_range();
  auto [result, face_im, point_im] = tf::reindexed_by_mask_on_points(
      poly, mask.make_range(), tf::return_index_map);
  return {
      wasm_mesh<Real>::from_polygons_buffer(std::move(result)),
      wasm_index_map::from_index_map_buffer(std::move(face_im)),
      wasm_index_map::from_index_map_buffer(std::move(point_im)),
  };
}

template <typename Real>
auto async_reindexed_by_mask_on_points(wasm_mesh<Real> &m,
                                       wasm_ndarray<std::int8_t> &mask)
    -> promise_t {
  return promise([a = m, b = mask]() -> wasm_mesh<Real> {
    return sync_reindexed_by_mask_on_points<Real>(
        const_cast<wasm_mesh<Real> &>(a),
        const_cast<wasm_ndarray<std::int8_t> &>(b));
  });
}

template <typename Real>
auto async_reindexed_by_mask_on_points_with_maps(
    wasm_mesh<Real> &m, wasm_ndarray<std::int8_t> &mask) -> promise_t {
  return promise([a = m, b = mask]() -> reindexed_mesh_result_t<Real> {
    return sync_reindexed_by_mask_on_points_with_maps<Real>(
        const_cast<wasm_mesh<Real> &>(a),
        const_cast<wasm_ndarray<std::int8_t> &>(b));
  });
}

// ============================================================================
// reindexed_by_ids_on_points(mesh, pointIds) ± maps
// ============================================================================

template <typename Real>
auto sync_reindexed_by_ids_on_points(wasm_mesh<Real> &m,
                                     wasm_ndarray<int> &ids) -> wasm_mesh<Real> {
  auto poly = m.polygons_range();
  auto [result, face_im, point_im] = tf::reindexed_by_ids_on_points(
      poly, ids.make_range(), tf::return_index_map);
  return wasm_mesh<Real>::from_polygons_buffer(std::move(result));
}

template <typename Real>
auto sync_reindexed_by_ids_on_points_with_maps(wasm_mesh<Real> &m,
                                               wasm_ndarray<int> &ids)
    -> reindexed_mesh_result_t<Real> {
  auto poly = m.polygons_range();
  auto [result, face_im, point_im] = tf::reindexed_by_ids_on_points(
      poly, ids.make_range(), tf::return_index_map);
  return {
      wasm_mesh<Real>::from_polygons_buffer(std::move(result)),
      wasm_index_map::from_index_map_buffer(std::move(face_im)),
      wasm_index_map::from_index_map_buffer(std::move(point_im)),
  };
}

template <typename Real>
auto async_reindexed_by_ids_on_points(wasm_mesh<Real> &m,
                                      wasm_ndarray<int> &ids) -> promise_t {
  return promise([a = m, b = ids]() -> wasm_mesh<Real> {
    return sync_reindexed_by_ids_on_points<Real>(
        const_cast<wasm_mesh<Real> &>(a), const_cast<wasm_ndarray<int> &>(b));
  });
}

template <typename Real>
auto async_reindexed_by_ids_on_points_with_maps(wasm_mesh<Real> &m,
                                                wasm_ndarray<int> &ids)
    -> promise_t {
  return promise([a = m, b = ids]() -> reindexed_mesh_result_t<Real> {
    return sync_reindexed_by_ids_on_points_with_maps<Real>(
        const_cast<wasm_mesh<Real> &>(a), const_cast<wasm_ndarray<int> &>(b));
  });
}

// ============================================================================
// concatenate_meshes — extract the JS mesh array on the main thread, then run
// the concatenation pipeline (optionally stitching per-mesh transformations).
// ============================================================================

template <typename Real>
auto extract_meshes(emscripten::val js_meshes) -> std::vector<wasm_mesh<Real>> {
  auto n = js_meshes["length"].as<int>();
  std::vector<wasm_mesh<Real>> meshes;
  meshes.reserve(n);
  for (int i = 0; i < n; ++i)
    meshes.push_back(
        js_meshes[i].as<wasm_mesh<Real>>(emscripten::allow_raw_pointers()));
  return meshes;
}

template <typename Real>
auto run_concatenate(std::vector<wasm_mesh<Real>> &meshes) -> wasm_mesh<Real> {
  bool any_transformed = false;
  for (auto &m : meshes)
    if (m.has_transformation())
      any_transformed = true;

  auto mesh_range = tf::make_range(meshes);

  if (any_transformed) {
    auto identity = tf::make_identity_transformation<Real, 3>();
    auto tagged = tf::make_mapped_range(mesh_range, [&](auto &m) {
      auto tv = m.has_transformation()
                    ? m.transformation_view()
                    : tf::make_transformation_view<3>(&identity(0, 0));
      return m.polygons_range() | tf::tag(tv);
    });
    return wasm_mesh<Real>::from_polygons_buffer(tf::concatenated(tagged));
  }

  auto plain = tf::make_mapped_range(mesh_range, [](auto &m) {
    return m.polygons_range();
  });
  return wasm_mesh<Real>::from_polygons_buffer(tf::concatenated(plain));
}

template <typename Real>
auto sync_concatenate_meshes(emscripten::val js_meshes) -> wasm_mesh<Real> {
  auto meshes = extract_meshes<Real>(js_meshes);
  if (meshes.empty())
    throw std::runtime_error("concatenateMeshes: empty array");
  return run_concatenate<Real>(meshes);
}

template <typename Real>
auto async_concatenate_meshes(emscripten::val js_meshes) -> promise_t {
  auto meshes = extract_meshes<Real>(js_meshes);
  return promise([ms = std::move(meshes)]() -> wasm_mesh<Real> {
    auto &meshes = const_cast<std::vector<wasm_mesh<Real>> &>(ms);
    return run_concatenate<Real>(meshes);
  });
}

// ============================================================================
// split_into_components(mesh, labels) — slice a mesh into one sub-mesh per
// connected-component label, returning the per-component unique label values.
// ============================================================================

template <typename Real>
auto compute_split(wasm_mesh<Real> &m, wasm_ndarray<int> &labels)
    -> split_components_result_t<Real> {
  auto poly = m.polygons_range();
  auto [components, comp_labels] =
      tf::split_into_components(poly, labels.make_range());

  split_components_result_t<Real> result;
  result.components.reserve(components.size());
  for (auto &c : components)
    result.components.push_back(
        wasm_mesh<Real>::from_polygons_buffer(std::move(c)));
  auto n_labels = static_cast<int>(comp_labels.size());
  result.labels =
      wasm_ndarray<int>::from_buffer(std::move(comp_labels), {n_labels});
  return result;
}

template <typename Real>
auto sync_split_into_components(wasm_mesh<Real> &m, wasm_ndarray<int> &labels)
    -> split_components_result_t<Real> {
  return compute_split<Real>(m, labels);
}

template <typename Real>
auto async_split_into_components(wasm_mesh<Real> &m, wasm_ndarray<int> &labels)
    -> promise_t {
  return promise([a = m, b = labels]() -> split_components_result_t<Real> {
    return compute_split<Real>(const_cast<wasm_mesh<Real> &>(a),
                               const_cast<wasm_ndarray<int> &>(b));
  });
}

} // namespace ts
} // namespace tf
