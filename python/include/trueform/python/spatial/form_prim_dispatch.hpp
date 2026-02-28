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
 * Author: Ziga Sajovic
 */

#pragma once

#include "trueform/python/core/prim_dispatch.hpp"
#include "trueform/python/spatial/form_intersects_primitive.hpp"
#include "trueform/python/spatial/gather_ids.hpp"
#include "trueform/python/spatial/neighbor_search.hpp"
#include "trueform/python/spatial/ray_cast.hpp"
#include "trueform/python/util/make_numpy_array.hpp"
#include "trueform/python/util/ray_config_helper.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <trueform/core/aabb_from.hpp>
#include <trueform/core/algorithm/parallel_fill.hpp>
#include <trueform/core/algorithm/parallel_for_each.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/distance.hpp>
#include <trueform/spatial/distance.hpp>
#include <trueform/core/intersects.hpp>
#include <trueform/core/sphere.hpp>
#include <trueform/core/views/sequence_range.hpp>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace tf::py {

// ============================================================================
// SFINAE trait: detect bounded primitives (those supporting tf::aabb_from)
// Bounded: point, segment, triangle, aabb, polygon
// Unbounded: ray, line, plane
// ============================================================================

namespace detail {

template <typename T, typename = void>
struct has_aabb_from : std::false_type {};

template <typename T>
struct has_aabb_from<
    T, std::void_t<decltype(tf::aabb_from(std::declval<const T &>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_aabb_from_v = has_aabb_from<T>::value;

} // namespace detail

// ============================================================================
// register_form_prim_ops: registration template shared by all per-type .cpp
// Registers 7 nanobind functions per form type combination:
//   Batch-capable (6): intersects, distance, distance2, ray_cast,
//                      neighbor_search, neighbor_search_knn
//   Single-only  (1): gather_ids
// ============================================================================

template <typename FormWrapper, std::size_t Dims, typename RealT>
auto register_form_prim_ops(nanobind::module_ &m, const char *form_name,
                            const char *suffix) -> void {
  namespace nb = nanobind;
  using PW = primitive_wrapper<Dims, RealT>;

  auto name = [form_name, suffix](const char *op) -> std::string {
    return std::string(op) + form_name + "_fp_" + suffix;
  };

  // ========================================================================
  // BATCH-CAPABLE OPS (single → scalar, batch → ndarray)
  // ========================================================================

  // ==== intersects: single → bool, batch → ndarray[int8] ====
  m.def(
      name("intersects_").c_str(),
      [](FormWrapper &fw, const PW &pw) -> nb::object {
        if (!pw.is_batch()) {
          auto hit = dispatch_single(
              [&](const auto &prim) -> bool {
                return form_intersects_primitive(fw, prim);
              },
              pw);
          return nb::cast(hit);
        }

        fw.tree();
        int n = pw.count();
        tf::buffer<std::int8_t> out;
        out.allocate(n);
        auto *dst = out.data();

        auto run = [&](const auto &form) {
          auto compute = [&](int i) {
            dst[i] = dispatch_at(
                         [&](const auto &prim) -> bool {
                           return tf::intersects(form, prim);
                         },
                         pw, i)
                         ? std::int8_t{1}
                         : std::int8_t{0};
          };
          if (n >= 1000)
            tf::parallel_for_each(tf::make_sequence_range(n),
                                  compute);
          else
            for (int i = 0; i < n; ++i)
              compute(i);
        };

        if (fw.has_transformation())
          run(fw.make_primitive_range() | tf::tag(fw.tree()) |
              tf::tag(
                  tf::make_frame(fw.transformation_view())));
        else
          run(fw.make_primitive_range() | tf::tag(fw.tree()));

        return nb::cast(make_numpy_array<nb::shape<-1>>(
            std::move(out), {static_cast<size_t>(n)}));
      },
      nb::arg("form"), nb::arg("primitive"));

  // ==== distance: single → float, batch → ndarray[float] ====
  m.def(
      name("distance_").c_str(),
      [](FormWrapper &fw, const PW &pw) -> nb::object {
        if (!pw.is_batch()) {
          auto run = [&](const auto &form) {
            return dispatch_single(
                [&](const auto &prim) -> RealT {
                  return tf::distance(form, prim);
                },
                pw);
          };
          RealT d;
          if (fw.has_transformation())
            d = run(fw.make_primitive_range() | tf::tag(fw.tree()) |
                    tf::tag(
                        tf::make_frame(fw.transformation_view())));
          else
            d = run(fw.make_primitive_range() | tf::tag(fw.tree()));
          return nb::cast(d);
        }

        fw.tree();
        int n = pw.count();
        tf::buffer<RealT> out;
        out.allocate(n);
        auto *dst = out.data();

        auto run = [&](const auto &form) {
          auto compute = [&](int i) {
            dispatch_at(
                [&](const auto &prim) {
                  dst[i] = tf::distance(form, prim);
                },
                pw, i);
          };
          if (n >= 1000)
            tf::parallel_for_each(tf::make_sequence_range(n),
                                  compute);
          else
            for (int i = 0; i < n; ++i)
              compute(i);
        };

        if (fw.has_transformation())
          run(fw.make_primitive_range() | tf::tag(fw.tree()) |
              tf::tag(
                  tf::make_frame(fw.transformation_view())));
        else
          run(fw.make_primitive_range() | tf::tag(fw.tree()));

        return nb::cast(make_numpy_array<nb::shape<-1>>(
            std::move(out), {static_cast<size_t>(n)}));
      },
      nb::arg("form"), nb::arg("primitive"));

  // ==== distance2: single → float, batch → ndarray[float] ====
  m.def(
      name("distance2_").c_str(),
      [](FormWrapper &fw, const PW &pw) -> nb::object {
        if (!pw.is_batch()) {
          auto run = [&](const auto &form) {
            return dispatch_single(
                [&](const auto &prim) -> RealT {
                  return tf::distance2(form, prim);
                },
                pw);
          };
          RealT d2;
          if (fw.has_transformation())
            d2 = run(fw.make_primitive_range() | tf::tag(fw.tree()) |
                     tf::tag(
                         tf::make_frame(fw.transformation_view())));
          else
            d2 = run(fw.make_primitive_range() | tf::tag(fw.tree()));
          return nb::cast(d2);
        }

        fw.tree();
        int n = pw.count();
        tf::buffer<RealT> out;
        out.allocate(n);
        auto *dst = out.data();

        auto run = [&](const auto &form) {
          auto compute = [&](int i) {
            dispatch_at(
                [&](const auto &prim) {
                  dst[i] = tf::distance2(form, prim);
                },
                pw, i);
          };
          if (n >= 1000)
            tf::parallel_for_each(tf::make_sequence_range(n),
                                  compute);
          else
            for (int i = 0; i < n; ++i)
              compute(i);
        };

        if (fw.has_transformation())
          run(fw.make_primitive_range() | tf::tag(fw.tree()) |
              tf::tag(
                  tf::make_frame(fw.transformation_view())));
        else
          run(fw.make_primitive_range() | tf::tag(fw.tree()));

        return nb::cast(make_numpy_array<nb::shape<-1>>(
            std::move(out), {static_cast<size_t>(n)}));
      },
      nb::arg("form"), nb::arg("primitive"));

  // ==== ray_cast: single → tuple(index, t)|None, batch → (ids[N], ts[N]) ====
  // config: optional pair of ndarrays (min_ts, max_ts). Python expands
  // scalar tuples and None to arrays before calling.
  using RayConfigArrays = std::pair<
      nb::ndarray<nb::numpy, RealT, nb::ndim<1>, nb::c_contig>,
      nb::ndarray<nb::numpy, RealT, nb::ndim<1>, nb::c_contig>>;
  m.def(
      name("ray_cast_").c_str(),
      [](const PW &ray_pw, FormWrapper &fw,
         std::optional<RayConfigArrays> opt_config) -> nb::object {
        if (!ray_pw.is_batch()) {
          // Single ray: config arrays have length 1
          std::optional<std::tuple<RealT, RealT>> scalar_config;
          if (opt_config)
            scalar_config = std::make_tuple(
                opt_config->first.data()[0], opt_config->second.data()[0]);
          auto ray = make_ray(ray_pw);
          auto result = ray_cast(ray, fw, scalar_config);
          if (result)
            return nb::cast(
                nb::make_tuple(result->first, result->second));
          return nb::none();
        }

        // Batch: Python always provides config arrays
        fw.tree();
        const RealT *min_ts = opt_config->first.data();
        const RealT *max_ts = opt_config->second.data();

        int n = ray_pw.count();
        using Index =
            typename std::decay_t<decltype(fw.tree())>::index_type;
        tf::buffer<Index> ids;
        ids.allocate(n);
        tf::buffer<RealT> ts;
        ts.allocate(n);
        auto *id_dst = ids.data();
        auto *t_dst = ts.data();
        constexpr auto nan = std::numeric_limits<RealT>::quiet_NaN();

        auto run = [&](const auto &form) {
          auto compute = [&](int i) {
            auto ray_ptr = ray_pw.element_ptr(i);
            auto pts = tf::make_points<Dims>(
                tf::make_range(ray_ptr, 2 * Dims));
            auto ray =
                tf::make_ray_like(pts[0], pts[1].as_vector_view());
            auto res = tf::ray_cast(
                ray, form,
                tf::ray_config<RealT>{min_ts[i], max_ts[i]});
            if (res) {
              id_dst[i] = res.element;
              t_dst[i] = static_cast<RealT>(res.info.t);
            } else {
              id_dst[i] = static_cast<Index>(-1);
              t_dst[i] = nan;
            }
          };
          if (n >= 1000)
            tf::parallel_for_each(tf::make_sequence_range(n),
                                  compute);
          else
            for (int i = 0; i < n; ++i)
              compute(i);
        };

        if (fw.has_transformation())
          run(fw.make_primitive_range() | tf::tag(fw.tree()) |
              tf::tag(
                  tf::make_frame(fw.transformation_view())));
        else
          run(fw.make_primitive_range() | tf::tag(fw.tree()));

        return nb::cast(nb::make_tuple(
            make_numpy_array<nb::shape<-1>>(
                std::move(ids), {static_cast<size_t>(n)}),
            make_numpy_array<nb::shape<-1>>(
                std::move(ts), {static_cast<size_t>(n)})));
      },
      nb::arg("ray"), nb::arg("form"),
      nb::arg("config").none() = nb::none());

  // ========================================================================
  // BATCH-CAPABLE NEIGHBOR SEARCH
  // (single → optional<tuple>/list<tuple>, batch → tuple of ndarrays)
  // ========================================================================

  // ==== neighbor_search (1-nn): single → tuple|None, batch →
  // (ids[N], dists[N], pts[N,D]) ====
  m.def(
      name("neighbor_search_").c_str(),
      [](FormWrapper &fw, const PW &pw,
         std::optional<RealT> radius) -> nb::object {
        if (!pw.is_batch()) {
          auto result = dispatch_single(
              [&](const auto &prim) {
                return neighbor_search<RealT, Dims>(fw, prim, radius);
              },
              pw);
          if (!result)
            return nb::none();
          return nb::cast(*result);
        }

        fw.tree();
        using Index =
            typename std::decay_t<decltype(fw.tree())>::index_type;
        int n = pw.count();
        RealT r =
            radius ? *radius : std::numeric_limits<RealT>::max();

        tf::buffer<Index> ids;
        ids.allocate(n);
        tf::buffer<RealT> dists;
        dists.allocate(n);
        tf::buffer<RealT> pts;
        pts.allocate(static_cast<std::size_t>(n) * Dims);
        auto *id_dst = ids.data();
        auto *dist_dst = dists.data();
        auto *pts_dst = pts.data();

        auto run = [&](const auto &form) {
          auto compute = [&](int i) {
            dispatch_at(
                [&](const auto &prim) {
                  auto res = tf::neighbor_search(form, prim, r);
                  if (res) {
                    id_dst[i] = res.element;
                    dist_dst[i] =
                        static_cast<RealT>(res.info.metric);
                    const auto &pt = res.info.point;
                    for (std::size_t d = 0; d < Dims; ++d)
                      pts_dst[i * Dims + d] =
                          static_cast<RealT>(pt[d]);
                  } else {
                    id_dst[i] = static_cast<Index>(-1);
                    dist_dst[i] =
                        std::numeric_limits<RealT>::infinity();
                    for (std::size_t d = 0; d < Dims; ++d)
                      pts_dst[i * Dims + d] = RealT{0};
                  }
                },
                pw, i);
          };
          if (n >= 100)
            tf::parallel_for_each(tf::make_sequence_range(n),
                                  compute);
          else
            for (int i = 0; i < n; ++i)
              compute(i);
        };

        if (fw.has_transformation())
          run(fw.make_primitive_range() | tf::tag(fw.tree()) |
              tf::tag(
                  tf::make_frame(fw.transformation_view())));
        else
          run(fw.make_primitive_range() | tf::tag(fw.tree()));

        return nb::cast(nb::make_tuple(
            make_numpy_array<nb::shape<-1>>(
                std::move(ids), {static_cast<size_t>(n)}),
            make_numpy_array<nb::shape<-1>>(
                std::move(dists), {static_cast<size_t>(n)}),
            make_numpy_array<nb::shape<-1, Dims>>(
                std::move(pts),
                {static_cast<size_t>(n), Dims})));
      },
      nb::arg("form"), nb::arg("primitive"),
      nb::arg("radius").none() = nb::none());

  // ==== neighbor_search (k-nn): single → list<tuple>, batch →
  // (ids[N,K], dists[N,K], pts[N,K,D], counts[N]) ====
  m.def(
      name("neighbor_search_knn_").c_str(),
      [](FormWrapper &fw, const PW &pw, int k,
         std::optional<RealT> radius) -> nb::object {
        if (!pw.is_batch()) {
          return nb::cast(dispatch_single(
              [&](const auto &prim) {
                return neighbor_search<RealT, Dims>(fw, prim, k,
                                                     radius);
              },
              pw));
        }

        fw.tree();
        using Index =
            typename std::decay_t<decltype(fw.tree())>::index_type;
        using nn_t = tf::nearest_neighbor<Index, RealT, Dims>;
        int n = pw.count();
        RealT r =
            radius ? *radius : std::numeric_limits<RealT>::max();

        std::size_t nk = static_cast<std::size_t>(n) * k;
        tf::buffer<Index> ids;
        ids.allocate(nk);
        tf::buffer<RealT> dists;
        dists.allocate(nk);
        tf::buffer<RealT> pts;
        pts.allocate(nk * Dims);
        tf::buffer<int> counts;
        counts.allocate(n);

        auto *id_dst = ids.data();
        auto *dist_dst = dists.data();
        auto *pts_dst = pts.data();
        auto *count_dst = counts.data();

        tf::parallel_fill(ids, static_cast<Index>(-1));

        auto run = [&](const auto &form) {
          std::vector<nn_t> state_buf;
          auto compute = [&](int i, std::vector<nn_t> &buf) {
            buf.resize(k);
            auto knn =
                tf::make_nearest_neighbors(buf.data(), k, r);
            dispatch_at(
                [&](const auto &prim) {
                  tf::neighbor_search(form, prim, knn);
                },
                pw, i);
            int cnt = static_cast<int>(knn.size());
            count_dst[i] = cnt;
            int base = i * k;
            for (int j = 0; j < cnt; ++j) {
              id_dst[base + j] = buf[j].element;
              dist_dst[base + j] =
                  static_cast<RealT>(buf[j].info.metric);
              for (std::size_t d = 0; d < Dims; ++d)
                pts_dst[(base + j) * Dims + d] =
                    static_cast<RealT>(buf[j].info.point[d]);
            }
          };
          if (n >= 100)
            tf::parallel_for_each(tf::make_sequence_range(n),
                                  compute,
                                  std::move(state_buf));
          else {
            state_buf.resize(k);
            for (int i = 0; i < n; ++i)
              compute(i, state_buf);
          }
        };

        if (fw.has_transformation())
          run(fw.make_primitive_range() | tf::tag(fw.tree()) |
              tf::tag(
                  tf::make_frame(fw.transformation_view())));
        else
          run(fw.make_primitive_range() | tf::tag(fw.tree()));

        return nb::cast(nb::make_tuple(
            make_numpy_array<nb::shape<-1, -1>>(
                std::move(ids),
                {static_cast<size_t>(n),
                 static_cast<size_t>(k)}),
            make_numpy_array<nb::shape<-1, -1>>(
                std::move(dists),
                {static_cast<size_t>(n),
                 static_cast<size_t>(k)}),
            make_numpy_array<nb::shape<-1, -1, Dims>>(
                std::move(pts),
                {static_cast<size_t>(n),
                 static_cast<size_t>(k), Dims}),
            make_numpy_array<nb::shape<-1>>(
                std::move(counts),
                {static_cast<size_t>(n)})));
      },
      nb::arg("form"), nb::arg("primitive"), nb::arg("k"),
      nb::arg("radius").none() = nb::none());

  // ========================================================================
  // SINGLE-ONLY OPS (no broadcasting)
  // ========================================================================

  // ==== gather_ids ====
  m.def(
      name("gather_ids_").c_str(),
      [](FormWrapper &fw, const PW &pw, const std::string &predicate_type,
         std::optional<RealT> threshold) {
        if (predicate_type != "intersects" &&
            predicate_type != "within_distance")
          throw std::runtime_error("Unknown predicate: " +
                                   predicate_type);
        if (predicate_type == "within_distance" && !threshold)
          throw std::runtime_error(
              "threshold required for within_distance");

        bool within = predicate_type == "within_distance";
        RealT threshold2 =
            within ? (*threshold) * (*threshold) : RealT{0};

        return dispatch_single(
            [&](const auto &query) {
              using Q = std::decay_t<decltype(query)>;

              if (!within) {
                if constexpr (detail::has_aabb_from_v<Q>) {
                  auto query_aabb = tf::aabb_from(query);
                  auto aabb_pred = [query_aabb](const auto &aabb) {
                    return tf::intersects(aabb, query_aabb);
                  };
                  auto prim_pred = [&query](const auto &prim) {
                    return tf::intersects(prim, query);
                  };
                  return gather_ids<RealT, Dims>(fw, aabb_pred,
                                                 prim_pred);
                } else {
                  auto aabb_pred = [&query](const auto &aabb) {
                    return tf::intersects(query, aabb);
                  };
                  auto prim_pred = [&query](const auto &prim) {
                    return tf::intersects(prim, query);
                  };
                  return gather_ids<RealT, Dims>(fw, aabb_pred,
                                                 prim_pred);
                }
              } else {
                if constexpr (detail::has_aabb_from_v<Q>) {
                  auto query_aabb = tf::aabb_from(query);
                  auto aabb_pred = [query_aabb,
                                    threshold2](const auto &aabb) {
                    return tf::distance2(aabb, query_aabb) <= threshold2;
                  };
                  auto prim_pred = [&query,
                                    threshold2](const auto &prim) {
                    return tf::distance2(prim, query) <= threshold2;
                  };
                  return gather_ids<RealT, Dims>(fw, aabb_pred,
                                                 prim_pred);
                } else {
                  auto aabb_pred = [&query,
                                    threshold2](const auto &aabb) {
                    return tf::distance2(
                               tf::make_sphere(
                                   aabb.center(),
                                   aabb.diagonal().length() / 2),
                               query) <= threshold2;
                  };
                  auto prim_pred = [&query,
                                    threshold2](const auto &prim) {
                    return tf::distance2(prim, query) <= threshold2;
                  };
                  return gather_ids<RealT, Dims>(fw, aabb_pred,
                                                 prim_pred);
                }
              }
            },
            pw);
      },
      nb::arg("form"), nb::arg("primitive"), nb::arg("predicate_type"),
      nb::arg("threshold").none() = nb::none());
}

} // namespace tf::py
