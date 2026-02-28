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

#include "trueform/core/frame.hpp"
#include "trueform/core/policy/frame.hpp"
#include "trueform/core/policy/normals.hpp"
#include "trueform/core/unit_vectors.hpp"
#include "trueform/geometry/chamfer_error.hpp"
#include "trueform/geometry/fit_icp_alignment.hpp"
#include "trueform/geometry/fit_obb_alignment.hpp"
#include "trueform/geometry/fit_rigid_alignment.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/to_wasm_matrix.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/core/wasm_point_cloud.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

// ============================================================================
// Frame dispatch helper
//
// Calls fn(tagged_src, tagged_tgt) with the correct frame tags.
// All 4 branches return the same type (transformation or float).
// ============================================================================

template <typename S, typename T, typename Fn>
auto dispatch_frames(wasm_point_cloud &src, wasm_point_cloud &tgt, S &&s, T &&t,
                     Fn &&fn) -> decltype(fn(s, t)) {
  bool h0 = src.has_transformation();
  bool h1 = tgt.has_transformation();
  if (h0 && h1)
    return fn(s | tf::tag(tf::make_frame(src.transformation_view())),
              t | tf::tag(tf::make_frame(tgt.transformation_view())));
  if (h0)
    return fn(s | tf::tag(tf::make_frame(src.transformation_view())), t);
  if (h1)
    return fn(s, t | tf::tag(tf::make_frame(tgt.transformation_view())));
  return fn(s, t);
}

// ============================================================================
// ICP alignment
// ============================================================================

auto sync_fit_icp(wasm_point_cloud &src, wasm_point_cloud &tgt,
                  int max_iterations, int n_samples, int k, float sigma,
                  float outlier_proportion, float min_relative_improvement,
                  float ema_alpha) -> emscripten::val {
  auto pts0 = src.points_range();
  auto pts1 = tgt.points_range();

  tf::icp_config config{static_cast<std::size_t>(max_iterations),
                        min_relative_improvement,
                        ema_alpha,
                        static_cast<std::size_t>(n_samples),
                        static_cast<std::size_t>(k),
                        sigma,
                        outlier_proportion};

  bool src_nrm = src.has_normals();
  bool tgt_nrm = tgt.has_normals();

  if (src_nrm && tgt_nrm) {
    auto n0 = tf::make_unit_vectors<3>(src.normals().make_range());
    auto n1 = tf::make_unit_vectors<3>(tgt.normals().make_range());
    return emscripten::val(to_wasm_matrix(dispatch_frames(
        src, tgt, pts0 | tf::tag_normals(n0),
        pts1 | tf::tag(tgt.tree()) | tf::tag_normals(n1),
        [&](auto &&s, auto &&t) {
          return tf::fit_icp_alignment(s, t, config);
        })));
  }
  if (tgt_nrm) {
    auto n1 = tf::make_unit_vectors<3>(tgt.normals().make_range());
    return emscripten::val(to_wasm_matrix(dispatch_frames(
        src, tgt, pts0, pts1 | tf::tag(tgt.tree()) | tf::tag_normals(n1),
        [&](auto &&s, auto &&t) {
          return tf::fit_icp_alignment(s, t, config);
        })));
  }
  return emscripten::val(to_wasm_matrix(dispatch_frames(
      src, tgt, pts0, pts1 | tf::tag(tgt.tree()),
      [&](auto &&s, auto &&t) {
        return tf::fit_icp_alignment(s, t, config);
      })));
}

auto async_fit_icp(wasm_point_cloud &src, wasm_point_cloud &tgt,
                   int max_iterations, int n_samples, int k, float sigma,
                   float outlier_proportion, float min_relative_improvement,
                   float ema_alpha) -> promise_t {
  // Ensure tree is built on the main thread before dispatching.
  tgt.ensure_tree();
  return promise(
      [src = src, tgt = tgt, max_iterations, n_samples, k, sigma,
       outlier_proportion, min_relative_improvement,
       ema_alpha]() -> wasm_ndarray<float> {
        auto &s = const_cast<wasm_point_cloud &>(src);
        auto &t = const_cast<wasm_point_cloud &>(tgt);
        auto pts0 = s.points_range();
        auto pts1 = t.points_range();

        tf::icp_config config{static_cast<std::size_t>(max_iterations),
                              min_relative_improvement,
                              ema_alpha,
                              static_cast<std::size_t>(n_samples),
                              static_cast<std::size_t>(k),
                              sigma,
                              outlier_proportion};

        bool src_nrm = s.has_normals();
        bool tgt_nrm = t.has_normals();

        if (src_nrm && tgt_nrm) {
          auto n0 = tf::make_unit_vectors<3>(s.normals().make_range());
          auto n1 = tf::make_unit_vectors<3>(t.normals().make_range());
          return to_wasm_matrix(dispatch_frames(
              s, t, pts0 | tf::tag_normals(n0),
              pts1 | tf::tag(t.tree()) | tf::tag_normals(n1),
              [&](auto &&s_, auto &&t_) {
                return tf::fit_icp_alignment(s_, t_, config);
              }));
        }
        if (tgt_nrm) {
          auto n1 = tf::make_unit_vectors<3>(t.normals().make_range());
          return to_wasm_matrix(dispatch_frames(
              s, t, pts0,
              pts1 | tf::tag(t.tree()) | tf::tag_normals(n1),
              [&](auto &&s_, auto &&t_) {
                return tf::fit_icp_alignment(s_, t_, config);
              }));
        }
        return to_wasm_matrix(dispatch_frames(
            s, t, pts0, pts1 | tf::tag(t.tree()),
            [&](auto &&s_, auto &&t_) {
              return tf::fit_icp_alignment(s_, t_, config);
            }));
      });
}

// ============================================================================
// Rigid alignment
// ============================================================================

auto sync_fit_rigid(wasm_point_cloud &src,
                    wasm_point_cloud &tgt) -> emscripten::val {
  auto pts0 = src.points_range();
  auto pts1 = tgt.points_range();

  bool src_nrm = src.has_normals();
  bool tgt_nrm = tgt.has_normals();

  if (src_nrm && tgt_nrm) {
    auto n0 = tf::make_unit_vectors<3>(src.normals().make_range());
    auto n1 = tf::make_unit_vectors<3>(tgt.normals().make_range());
    return emscripten::val(to_wasm_matrix(dispatch_frames(
        src, tgt, pts0 | tf::tag_normals(n0), pts1 | tf::tag_normals(n1),
        [&](auto &&s, auto &&t) { return tf::fit_rigid_alignment(s, t); })));
  }
  if (tgt_nrm) {
    auto n1 = tf::make_unit_vectors<3>(tgt.normals().make_range());
    return emscripten::val(to_wasm_matrix(dispatch_frames(
        src, tgt, pts0, pts1 | tf::tag_normals(n1),
        [&](auto &&s, auto &&t) { return tf::fit_rigid_alignment(s, t); })));
  }
  return emscripten::val(to_wasm_matrix(dispatch_frames(
      src, tgt, pts0, pts1,
      [&](auto &&s, auto &&t) { return tf::fit_rigid_alignment(s, t); })));
}

auto async_fit_rigid(wasm_point_cloud &src,
                     wasm_point_cloud &tgt) -> promise_t {
  return promise(
      [src = src, tgt = tgt]() -> wasm_ndarray<float> {
        auto &s = const_cast<wasm_point_cloud &>(src);
        auto &t = const_cast<wasm_point_cloud &>(tgt);
        auto pts0 = s.points_range();
        auto pts1 = t.points_range();

        bool src_nrm = s.has_normals();
        bool tgt_nrm = t.has_normals();

        if (src_nrm && tgt_nrm) {
          auto n0 = tf::make_unit_vectors<3>(s.normals().make_range());
          auto n1 = tf::make_unit_vectors<3>(t.normals().make_range());
          return to_wasm_matrix(dispatch_frames(
              s, t, pts0 | tf::tag_normals(n0), pts1 | tf::tag_normals(n1),
              [&](auto &&s_, auto &&t_) {
                return tf::fit_rigid_alignment(s_, t_);
              }));
        }
        if (tgt_nrm) {
          auto n1 = tf::make_unit_vectors<3>(t.normals().make_range());
          return to_wasm_matrix(dispatch_frames(
              s, t, pts0, pts1 | tf::tag_normals(n1),
              [&](auto &&s_, auto &&t_) {
                return tf::fit_rigid_alignment(s_, t_);
              }));
        }
        return to_wasm_matrix(dispatch_frames(
            s, t, pts0, pts1, [&](auto &&s_, auto &&t_) {
              return tf::fit_rigid_alignment(s_, t_);
            }));
      });
}

// ============================================================================
// OBB alignment (no normals)
// ============================================================================

auto sync_fit_obb(wasm_point_cloud &src, wasm_point_cloud &tgt,
                  int sample_size) -> emscripten::val {
  auto pts0 = src.points_range();
  auto pts1 = tgt.points_range();
  auto sz = static_cast<std::size_t>(sample_size);

  return emscripten::val(to_wasm_matrix(dispatch_frames(
      src, tgt, pts0, pts1 | tf::tag(tgt.tree()),
      [&](auto &&s, auto &&t) {
        return tf::fit_obb_alignment(s, t, sz);
      })));
}

auto async_fit_obb(wasm_point_cloud &src, wasm_point_cloud &tgt,
                   int sample_size) -> promise_t {
  tgt.ensure_tree();
  return promise(
      [src = src, tgt = tgt, sample_size]() -> wasm_ndarray<float> {
        auto &s = const_cast<wasm_point_cloud &>(src);
        auto &t = const_cast<wasm_point_cloud &>(tgt);
        auto pts0 = s.points_range();
        auto pts1 = t.points_range();
        auto sz = static_cast<std::size_t>(sample_size);
        return to_wasm_matrix(dispatch_frames(
            s, t, pts0, pts1 | tf::tag(t.tree()),
            [&](auto &&s_, auto &&t_) {
              return tf::fit_obb_alignment(s_, t_, sz);
            }));
      });
}

// ============================================================================
// Chamfer error (no normals, returns float)
// ============================================================================

auto sync_chamfer_error(wasm_point_cloud &src, wasm_point_cloud &tgt,
                        float outlier_proportion) -> float {
  auto pts0 = src.points_range();
  auto pts1 = tgt.points_range();

  return dispatch_frames(src, tgt, pts0, pts1 | tf::tag(tgt.tree()),
                         [&](auto &&s, auto &&t) {
                           return tf::chamfer_error(s, t, outlier_proportion);
                         });
}

auto async_chamfer_error(wasm_point_cloud &src, wasm_point_cloud &tgt,
                         float outlier_proportion) -> promise_t {
  tgt.ensure_tree();
  return promise(
      [src = src, tgt = tgt, outlier_proportion]() -> float {
        auto &s = const_cast<wasm_point_cloud &>(src);
        auto &t = const_cast<wasm_point_cloud &>(tgt);
        auto pts0 = s.points_range();
        auto pts1 = t.points_range();
        return dispatch_frames(
            s, t, pts0, pts1 | tf::tag(t.tree()),
            [&](auto &&s_, auto &&t_) {
              return tf::chamfer_error(s_, t_, outlier_proportion);
            });
      });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_registration) {
  emscripten::function("fit_icp_alignment", &sync_fit_icp);
  emscripten::function("fit_rigid_alignment", &sync_fit_rigid);
  emscripten::function("fit_obb_alignment", &sync_fit_obb);
  emscripten::function("chamfer_error", &sync_chamfer_error);
  emscripten::function("dispatch_fit_icp_alignment", &async_fit_icp);
  emscripten::function("dispatch_fit_rigid_alignment", &async_fit_rigid);
  emscripten::function("dispatch_fit_obb_alignment", &async_fit_obb);
  emscripten::function("dispatch_chamfer_error", &async_chamfer_error);
}
