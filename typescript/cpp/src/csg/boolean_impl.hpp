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

#include "trueform/csg/make_boolean.hpp"
#include "trueform/ts/core/build_intersect_structures.hpp"
#include "trueform/ts/core/extract_int_vector.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_curves.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <cstdint>
#include <vector>

namespace tf {
namespace ts {

// ============================================================================
// Result types — output mesh + (optional) intersection curves carry the
// inputs' coordinate dtype; labels stay int8 / int32.
// ============================================================================

template <typename Real> struct labeled_boolean_result_t {
  wasm_mesh<Real> mesh;
  wasm_ndarray<std::int8_t> labels;
  wasm_ndarray<std::int32_t> face_labels;
};

template <typename Real> struct labeled_boolean_result_with_curves_t {
  wasm_mesh<Real> mesh;
  wasm_ndarray<std::int8_t> labels;
  wasm_ndarray<std::int32_t> face_labels;
  wasm_curves<Real> curves;
};

// ============================================================================
// Sync — driver routines parameterized on `tf::boolean_op`. The trueform
// boolean pipeline is templated end-to-end on Real; here we just stitch in
// per-mesh transformations and rebuild the wasm-side result.
// ============================================================================

template <typename Real>
auto sync_boolean(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1, tf::boolean_op op,
                  const std::vector<int> &sheets)
    -> labeled_boolean_result_t<Real> {
  bool has0 = m0.has_transformation();
  bool has1 = m1.has_transformation();
  build_intersect_structures(m0, m1);
  auto fm0 = m0.face_membership_range();
  auto mel0 = m0.manifold_edge_link_range();
  auto fm1 = m1.face_membership_range();
  auto mel1 = m1.manifold_edge_link_range();
  auto make_return = [&](auto &&poly0,
                         auto &&poly1) -> labeled_boolean_result_t<Real> {
    auto [poly, labels, face_labels] = tf::make_boolean(
        poly0 | tf::tag(m0.tree()) | tf::tag(fm0) | tf::tag(mel0),
        poly1 | tf::tag(m1.tree()) | tf::tag(fm1) | tf::tag(mel1), op,
        tf::make_range(sheets.data(), sheets.data() + sheets.size()));
    return {wasm_mesh<Real>::from_polygons_buffer(std::move(poly)),
            wasm_ndarray<std::int8_t>::from_buffer(std::move(labels)),
            wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels))};
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
                       m1.polygons_range() |
                           tf::tag(m1.transformation_view()));
  else
    return make_return(m0.polygons_range(), m1.polygons_range());
}

template <typename Real>
auto sync_boolean_with_curves(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1,
                              tf::boolean_op op,
                              const std::vector<int> &sheets)
    -> labeled_boolean_result_with_curves_t<Real> {
  bool has0 = m0.has_transformation();
  bool has1 = m1.has_transformation();
  build_intersect_structures(m0, m1);
  auto fm0 = m0.face_membership_range();
  auto mel0 = m0.manifold_edge_link_range();
  auto fm1 = m1.face_membership_range();
  auto mel1 = m1.manifold_edge_link_range();
  auto make_return =
      [&](auto &&poly0,
          auto &&poly1) -> labeled_boolean_result_with_curves_t<Real> {
    auto [poly, labels, face_labels, curves] = tf::make_boolean(
        poly0 | tf::tag(m0.tree()) | tf::tag(fm0) | tf::tag(mel0),
        poly1 | tf::tag(m1.tree()) | tf::tag(fm1) | tf::tag(mel1),
        op, tf::make_range(sheets.data(), sheets.data() + sheets.size()),
        tf::return_curves);
    return {wasm_mesh<Real>::from_polygons_buffer(std::move(poly)),
            wasm_ndarray<std::int8_t>::from_buffer(std::move(labels)),
            wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels)),
            wasm_curves<Real>::from_curves_buffer(std::move(curves))};
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
                       m1.polygons_range() |
                           tf::tag(m1.transformation_view()));
  else
    return make_return(m0.polygons_range(), m1.polygons_range());
}

// -- Sync wrappers per operation --

template <typename Real>
auto sync_boolean_union(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1,
                    emscripten::val js_sheets)
    -> labeled_boolean_result_t<Real> {
  return sync_boolean<Real>(m0, m1, tf::boolean_op::merge,
                            extract_int_vector(js_sheets));
}

template <typename Real>
auto sync_boolean_intersection(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1,
                    emscripten::val js_sheets)
    -> labeled_boolean_result_t<Real> {
  return sync_boolean<Real>(m0, m1, tf::boolean_op::intersection,
                            extract_int_vector(js_sheets));
}

template <typename Real>
auto sync_boolean_difference(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1,
                    emscripten::val js_sheets)
    -> labeled_boolean_result_t<Real> {
  return sync_boolean<Real>(m0, m1, tf::boolean_op::left_difference,
                            extract_int_vector(js_sheets));
}

template <typename Real>
auto sync_boolean_union_with_curves(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1,
                    emscripten::val js_sheets)
    -> labeled_boolean_result_with_curves_t<Real> {
  return sync_boolean_with_curves<Real>(m0, m1, tf::boolean_op::merge,
                            extract_int_vector(js_sheets));
}

template <typename Real>
auto sync_boolean_intersection_with_curves(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1,
                    emscripten::val js_sheets)
    -> labeled_boolean_result_with_curves_t<Real> {
  return sync_boolean_with_curves<Real>(m0, m1, tf::boolean_op::intersection,
                            extract_int_vector(js_sheets));
}

template <typename Real>
auto sync_boolean_difference_with_curves(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1,
                    emscripten::val js_sheets)
    -> labeled_boolean_result_with_curves_t<Real> {
  return sync_boolean_with_curves<Real>(m0, m1, tf::boolean_op::left_difference,
                            extract_int_vector(js_sheets));
}

// ============================================================================
// Async — capture by copy (shared_ptr refcount bump) and const_cast through
// to the sync entry point on the worker thread.
// ============================================================================

template <typename Real>
auto async_boolean_union(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1,
                    emscripten::val js_sheets) -> promise_t {
  auto s = extract_int_vector(js_sheets);
  return promise([a = m0, b = m1, s = std::move(s)]() -> labeled_boolean_result_t<Real> {
    return sync_boolean<Real>(const_cast<wasm_mesh<Real> &>(a),
                              const_cast<wasm_mesh<Real> &>(b),
                              tf::boolean_op::merge, s);
  });
}

template <typename Real>
auto async_boolean_intersection(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1,
                    emscripten::val js_sheets) -> promise_t {
  auto s = extract_int_vector(js_sheets);
  return promise([a = m0, b = m1, s = std::move(s)]() -> labeled_boolean_result_t<Real> {
    return sync_boolean<Real>(const_cast<wasm_mesh<Real> &>(a),
                              const_cast<wasm_mesh<Real> &>(b),
                              tf::boolean_op::intersection, s);
  });
}

template <typename Real>
auto async_boolean_difference(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1,
                    emscripten::val js_sheets) -> promise_t {
  auto s = extract_int_vector(js_sheets);
  return promise([a = m0, b = m1, s = std::move(s)]() -> labeled_boolean_result_t<Real> {
    return sync_boolean<Real>(const_cast<wasm_mesh<Real> &>(a),
                              const_cast<wasm_mesh<Real> &>(b),
                              tf::boolean_op::left_difference, s);
  });
}

template <typename Real>
auto async_boolean_union_with_curves(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1,
                    emscripten::val js_sheets) -> promise_t {
  auto s = extract_int_vector(js_sheets);
  return promise([a = m0, b = m1, s = std::move(s)]() -> labeled_boolean_result_with_curves_t<Real> {
    return sync_boolean_with_curves<Real>(
        const_cast<wasm_mesh<Real> &>(a), const_cast<wasm_mesh<Real> &>(b),
        tf::boolean_op::merge, s);
  });
}

template <typename Real>
auto async_boolean_intersection_with_curves(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1,
                    emscripten::val js_sheets) -> promise_t {
  auto s = extract_int_vector(js_sheets);
  return promise([a = m0, b = m1, s = std::move(s)]() -> labeled_boolean_result_with_curves_t<Real> {
    return sync_boolean_with_curves<Real>(
        const_cast<wasm_mesh<Real> &>(a), const_cast<wasm_mesh<Real> &>(b),
        tf::boolean_op::intersection, s);
  });
}

template <typename Real>
auto async_boolean_difference_with_curves(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1,
                    emscripten::val js_sheets) -> promise_t {
  auto s = extract_int_vector(js_sheets);
  return promise([a = m0, b = m1, s = std::move(s)]() -> labeled_boolean_result_with_curves_t<Real> {
    return sync_boolean_with_curves<Real>(
        const_cast<wasm_mesh<Real> &>(a), const_cast<wasm_mesh<Real> &>(b),
        tf::boolean_op::left_difference, s);
  });
}

} // namespace ts
} // namespace tf
