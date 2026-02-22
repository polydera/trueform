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

#include "trueform/cut/make_boolean.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/cut/result_types.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

// -- Sync --

auto sync_boolean(wasm_mesh &m0, wasm_mesh &m1, tf::boolean_op op)
    -> labeled_cut_result {
  bool has0 = m0.has_transformation();
  bool has1 = m1.has_transformation();
  auto make_return = [&](auto &&poly0, auto &&poly1) -> labeled_cut_result {
    auto [poly, labels] = tf::make_boolean(poly0, poly1, op);
    return {wasm_mesh::from_polygons_buffer(std::move(poly)),
            wasm_ndarray<std::int8_t>::from_buffer(std::move(labels))};
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

auto sync_boolean_with_curves(wasm_mesh &m0, wasm_mesh &m1, tf::boolean_op op)
    -> labeled_cut_result_with_curves {
  bool has0 = m0.has_transformation();
  bool has1 = m1.has_transformation();
  auto make_return = [&](auto &&poly0,
                         auto &&poly1) -> labeled_cut_result_with_curves {
    auto [poly, labels, curves] =
        tf::make_boolean(poly0, poly1, op, tf::return_curves);
    return {wasm_mesh::from_polygons_buffer(std::move(poly)),
            wasm_ndarray<std::int8_t>::from_buffer(std::move(labels)),
            wasm_curves::from_curves_buffer(std::move(curves))};
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

auto sync_boolean_union(wasm_mesh &m0, wasm_mesh &m1) -> labeled_cut_result {
  return sync_boolean(m0, m1, tf::boolean_op::merge);
}
auto sync_boolean_intersection(wasm_mesh &m0, wasm_mesh &m1)
    -> labeled_cut_result {
  return sync_boolean(m0, m1, tf::boolean_op::intersection);
}
auto sync_boolean_difference(wasm_mesh &m0, wasm_mesh &m1)
    -> labeled_cut_result {
  return sync_boolean(m0, m1, tf::boolean_op::left_difference);
}

auto sync_boolean_union_with_curves(wasm_mesh &m0, wasm_mesh &m1)
    -> labeled_cut_result_with_curves {
  return sync_boolean_with_curves(m0, m1, tf::boolean_op::merge);
}
auto sync_boolean_intersection_with_curves(wasm_mesh &m0, wasm_mesh &m1)
    -> labeled_cut_result_with_curves {
  return sync_boolean_with_curves(m0, m1, tf::boolean_op::intersection);
}
auto sync_boolean_difference_with_curves(wasm_mesh &m0, wasm_mesh &m1)
    -> labeled_cut_result_with_curves {
  return sync_boolean_with_curves(m0, m1, tf::boolean_op::left_difference);
}

// -- Async --

auto async_boolean_union(wasm_mesh &m0, wasm_mesh &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> labeled_cut_result {
    return sync_boolean_union(const_cast<wasm_mesh &>(a),
                              const_cast<wasm_mesh &>(b));
  });
}
auto async_boolean_intersection(wasm_mesh &m0, wasm_mesh &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> labeled_cut_result {
    return sync_boolean_intersection(const_cast<wasm_mesh &>(a),
                                     const_cast<wasm_mesh &>(b));
  });
}
auto async_boolean_difference(wasm_mesh &m0, wasm_mesh &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> labeled_cut_result {
    return sync_boolean_difference(const_cast<wasm_mesh &>(a),
                                   const_cast<wasm_mesh &>(b));
  });
}

auto async_boolean_union_with_curves(wasm_mesh &m0, wasm_mesh &m1)
    -> promise_t {
  return promise(
      [a = m0, b = m1]() -> labeled_cut_result_with_curves {
        return sync_boolean_union_with_curves(const_cast<wasm_mesh &>(a),
                                              const_cast<wasm_mesh &>(b));
      });
}
auto async_boolean_intersection_with_curves(wasm_mesh &m0, wasm_mesh &m1)
    -> promise_t {
  return promise(
      [a = m0, b = m1]() -> labeled_cut_result_with_curves {
        return sync_boolean_intersection_with_curves(
            const_cast<wasm_mesh &>(a), const_cast<wasm_mesh &>(b));
      });
}
auto async_boolean_difference_with_curves(wasm_mesh &m0, wasm_mesh &m1)
    -> promise_t {
  return promise(
      [a = m0, b = m1]() -> labeled_cut_result_with_curves {
        return sync_boolean_difference_with_curves(
            const_cast<wasm_mesh &>(a), const_cast<wasm_mesh &>(b));
      });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_boolean) {
  // Result types
  emscripten::value_object<tf::ts::labeled_cut_result>("LabeledCutResult")
      .field("mesh", &tf::ts::labeled_cut_result::mesh)
      .field("labels", &tf::ts::labeled_cut_result::labels);

  emscripten::value_object<tf::ts::labeled_cut_result_with_curves>(
      "LabeledCutResultWithCurves")
      .field("mesh", &tf::ts::labeled_cut_result_with_curves::mesh)
      .field("labels", &tf::ts::labeled_cut_result_with_curves::labels)
      .field("curves", &tf::ts::labeled_cut_result_with_curves::curves);

  // Sync
  emscripten::function("boolean_union", &sync_boolean_union);
  emscripten::function("boolean_intersection", &sync_boolean_intersection);
  emscripten::function("boolean_difference", &sync_boolean_difference);
  emscripten::function("boolean_union_with_curves",
                       &sync_boolean_union_with_curves);
  emscripten::function("boolean_intersection_with_curves",
                       &sync_boolean_intersection_with_curves);
  emscripten::function("boolean_difference_with_curves",
                       &sync_boolean_difference_with_curves);

  // Async
  emscripten::function("dispatch_boolean_union", &async_boolean_union);
  emscripten::function("dispatch_boolean_intersection",
                       &async_boolean_intersection);
  emscripten::function("dispatch_boolean_difference",
                       &async_boolean_difference);
  emscripten::function("dispatch_boolean_union_with_curves",
                       &async_boolean_union_with_curves);
  emscripten::function("dispatch_boolean_intersection_with_curves",
                       &async_boolean_intersection_with_curves);
  emscripten::function("dispatch_boolean_difference_with_curves",
                       &async_boolean_difference_with_curves);
}
