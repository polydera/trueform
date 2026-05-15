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

namespace tf {
namespace ts {

// ============================================================================
// Frame dispatch helper
//
// Calls fn(tagged_src, tagged_tgt) with the correct frame tags.
// All 4 branches return the same type (transformation or scalar).
// ============================================================================

template <typename Real, typename S, typename T, typename Fn>
auto dispatch_frames(wasm_point_cloud<Real> &src, wasm_point_cloud<Real> &tgt,
                     S &&s, T &&t, Fn &&fn) -> decltype(fn(s, t)) {
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

template <typename Real>
auto sync_fit_icp(wasm_point_cloud<Real> &src, wasm_point_cloud<Real> &tgt,
                  int max_iterations, int n_samples, int k, Real sigma,
                  Real outlier_proportion, Real min_relative_improvement,
                  Real ema_alpha) -> wasm_ndarray<Real> {
  auto pts0 = src.points_range();
  auto pts1 = tgt.points_range();

  tf::icp_config config{static_cast<std::size_t>(max_iterations),
                        static_cast<float>(min_relative_improvement),
                        static_cast<float>(ema_alpha),
                        static_cast<std::size_t>(n_samples),
                        static_cast<std::size_t>(k),
                        static_cast<float>(sigma),
                        static_cast<float>(outlier_proportion)};

  bool src_nrm = src.has_normals();
  bool tgt_nrm = tgt.has_normals();

  if (src_nrm && tgt_nrm) {
    auto n0 = tf::make_unit_vectors<3>(src.normals().make_range());
    auto n1 = tf::make_unit_vectors<3>(tgt.normals().make_range());
    return to_wasm_matrix<Real>(dispatch_frames<Real>(
        src, tgt, pts0 | tf::tag_normals(n0),
        pts1 | tf::tag(tgt.tree()) | tf::tag_normals(n1),
        [&](auto &&s, auto &&t) {
          return tf::fit_icp_alignment(s, t, config);
        }));
  }
  if (tgt_nrm) {
    auto n1 = tf::make_unit_vectors<3>(tgt.normals().make_range());
    return to_wasm_matrix<Real>(dispatch_frames<Real>(
        src, tgt, pts0, pts1 | tf::tag(tgt.tree()) | tf::tag_normals(n1),
        [&](auto &&s, auto &&t) {
          return tf::fit_icp_alignment(s, t, config);
        }));
  }
  return to_wasm_matrix<Real>(dispatch_frames<Real>(
      src, tgt, pts0, pts1 | tf::tag(tgt.tree()),
      [&](auto &&s, auto &&t) {
        return tf::fit_icp_alignment(s, t, config);
      }));
}

template <typename Real>
auto async_fit_icp(wasm_point_cloud<Real> &src, wasm_point_cloud<Real> &tgt,
                   int max_iterations, int n_samples, int k, Real sigma,
                   Real outlier_proportion, Real min_relative_improvement,
                   Real ema_alpha) -> promise_t {
  // Ensure tree is built on the main thread before dispatching.
  (void)tgt.tree();
  return promise(
      [src = src, tgt = tgt, max_iterations, n_samples, k, sigma,
       outlier_proportion, min_relative_improvement,
       ema_alpha]() -> wasm_ndarray<Real> {
        return sync_fit_icp<Real>(
            const_cast<wasm_point_cloud<Real> &>(src),
            const_cast<wasm_point_cloud<Real> &>(tgt),
            max_iterations, n_samples, k, sigma, outlier_proportion,
            min_relative_improvement, ema_alpha);
      });
}

// ============================================================================
// Rigid alignment
// ============================================================================

template <typename Real>
auto sync_fit_rigid(wasm_point_cloud<Real> &src,
                    wasm_point_cloud<Real> &tgt) -> wasm_ndarray<Real> {
  auto pts0 = src.points_range();
  auto pts1 = tgt.points_range();

  bool src_nrm = src.has_normals();
  bool tgt_nrm = tgt.has_normals();

  if (src_nrm && tgt_nrm) {
    auto n0 = tf::make_unit_vectors<3>(src.normals().make_range());
    auto n1 = tf::make_unit_vectors<3>(tgt.normals().make_range());
    return to_wasm_matrix<Real>(dispatch_frames<Real>(
        src, tgt, pts0 | tf::tag_normals(n0), pts1 | tf::tag_normals(n1),
        [&](auto &&s, auto &&t) { return tf::fit_rigid_alignment(s, t); }));
  }
  if (tgt_nrm) {
    auto n1 = tf::make_unit_vectors<3>(tgt.normals().make_range());
    return to_wasm_matrix<Real>(dispatch_frames<Real>(
        src, tgt, pts0, pts1 | tf::tag_normals(n1),
        [&](auto &&s, auto &&t) { return tf::fit_rigid_alignment(s, t); }));
  }
  return to_wasm_matrix<Real>(dispatch_frames<Real>(
      src, tgt, pts0, pts1,
      [&](auto &&s, auto &&t) { return tf::fit_rigid_alignment(s, t); }));
}

template <typename Real>
auto async_fit_rigid(wasm_point_cloud<Real> &src,
                     wasm_point_cloud<Real> &tgt) -> promise_t {
  return promise([src = src, tgt = tgt]() -> wasm_ndarray<Real> {
    return sync_fit_rigid<Real>(const_cast<wasm_point_cloud<Real> &>(src),
                                const_cast<wasm_point_cloud<Real> &>(tgt));
  });
}

// ============================================================================
// OBB alignment (no normals)
// ============================================================================

template <typename Real>
auto sync_fit_obb(wasm_point_cloud<Real> &src, wasm_point_cloud<Real> &tgt,
                  int sample_size) -> wasm_ndarray<Real> {
  auto pts0 = src.points_range();
  auto pts1 = tgt.points_range();
  auto sz = static_cast<std::size_t>(sample_size);

  return to_wasm_matrix<Real>(dispatch_frames<Real>(
      src, tgt, pts0, pts1 | tf::tag(tgt.tree()),
      [&](auto &&s, auto &&t) {
        return tf::fit_obb_alignment(s, t, sz);
      }));
}

template <typename Real>
auto async_fit_obb(wasm_point_cloud<Real> &src, wasm_point_cloud<Real> &tgt,
                   int sample_size) -> promise_t {
  (void)tgt.tree();
  return promise([src = src, tgt = tgt, sample_size]() -> wasm_ndarray<Real> {
    return sync_fit_obb<Real>(const_cast<wasm_point_cloud<Real> &>(src),
                              const_cast<wasm_point_cloud<Real> &>(tgt),
                              sample_size);
  });
}

// ============================================================================
// Chamfer error (no normals, returns scalar)
// ============================================================================

template <typename Real>
auto sync_chamfer_error(wasm_point_cloud<Real> &src,
                        wasm_point_cloud<Real> &tgt,
                        Real outlier_proportion) -> Real {
  auto pts0 = src.points_range();
  auto pts1 = tgt.points_range();

  return dispatch_frames<Real>(
      src, tgt, pts0, pts1 | tf::tag(tgt.tree()),
      [&](auto &&s, auto &&t) {
        return tf::chamfer_error(s, t, outlier_proportion);
      });
}

template <typename Real>
auto async_chamfer_error(wasm_point_cloud<Real> &src,
                         wasm_point_cloud<Real> &tgt,
                         Real outlier_proportion) -> promise_t {
  (void)tgt.tree();
  return promise(
      [src = src, tgt = tgt, outlier_proportion]() -> Real {
        return sync_chamfer_error<Real>(
            const_cast<wasm_point_cloud<Real> &>(src),
            const_cast<wasm_point_cloud<Real> &>(tgt), outlier_proportion);
      });
}

} // namespace ts
} // namespace tf
