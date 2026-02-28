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

#include "trueform/python/core/prim_dispatch.hpp"
#include "trueform/python/util/make_numpy_array.hpp"
#include "trueform/python/util/ray_config_helper.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <trueform/core/algorithm/parallel_for_each.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/closest_metric_point_pair.hpp>
#include <trueform/core/distance.hpp>
#include <trueform/core/intersects.hpp>
#include <trueform/core/ray_cast.hpp>
#include <trueform/core/ray_config.hpp>
#include <trueform/core/views/sequence_range.hpp>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>

namespace tf::py {

namespace {

// ============================================================================
// Name helper
// ============================================================================

auto fn_name(const char *base, const char *suffix) -> std::string {
  return std::string(base) + suffix;
}

// ============================================================================
// Batch size helper
// ============================================================================

template <std::size_t Dims, typename RealT>
auto batch_count_pair(const primitive_wrapper<Dims, RealT> &a,
                      const primitive_wrapper<Dims, RealT> &b) -> int {
  bool ba = a.is_batch();
  bool bb = b.is_batch();
  int n = ba ? a.count() : b.count();
  if (ba && bb && a.count() != b.count())
    throw std::runtime_error("batch size mismatch");
  return n;
}

// ============================================================================
// Registration template
// ============================================================================

template <std::size_t Dims, typename RealT>
auto register_prim_prim_ops_impl(nanobind::module_ &m, const char *suffix)
    -> void {
  namespace nb = nanobind;
  using PW = primitive_wrapper<Dims, RealT>;

  auto name = [suffix](const char *base) -> std::string {
    return fn_name(base, suffix);
  };

  // ==== distance ====
  m.def(
      name("distance_pp_").c_str(),
      [](const PW &a, const PW &b) -> nb::object {
        if (!a.is_batch() && !b.is_batch()) {
          auto d = dispatch_pair(
              [](const auto &pa, const auto &pb) -> RealT {
                return static_cast<RealT>(tf::distance(pa, pb));
              },
              a, b);
          return nb::cast(d);
        }

        int n = batch_count_pair(a, b);
        tf::buffer<RealT> out;
        out.allocate(n);
        auto *dst = out.data();

        auto compute = [&](int i) {
          dst[i] = static_cast<RealT>(dispatch_pair_at(
              [](const auto &pa, const auto &pb) -> RealT {
                return static_cast<RealT>(tf::distance(pa, pb));
              },
              a, i, b, i));
        };

        if (n >= 5000)
          tf::parallel_for_each(tf::make_sequence_range(n), compute);
        else
          for (int i = 0; i < n; ++i)
            compute(i);

        return nb::cast(make_numpy_array<nb::shape<-1>>(
            std::move(out), {static_cast<size_t>(n)}));
      },
      nb::arg("a"), nb::arg("b"));

  // ==== distance2 ====
  m.def(
      name("distance2_pp_").c_str(),
      [](const PW &a, const PW &b) -> nb::object {
        if (!a.is_batch() && !b.is_batch()) {
          auto d = dispatch_pair(
              [](const auto &pa, const auto &pb) -> RealT {
                return static_cast<RealT>(tf::distance2(pa, pb));
              },
              a, b);
          return nb::cast(d);
        }

        int n = batch_count_pair(a, b);
        tf::buffer<RealT> out;
        out.allocate(n);
        auto *dst = out.data();

        auto compute = [&](int i) {
          dst[i] = static_cast<RealT>(dispatch_pair_at(
              [](const auto &pa, const auto &pb) -> RealT {
                return static_cast<RealT>(tf::distance2(pa, pb));
              },
              a, i, b, i));
        };

        if (n >= 5000)
          tf::parallel_for_each(tf::make_sequence_range(n), compute);
        else
          for (int i = 0; i < n; ++i)
            compute(i);

        return nb::cast(make_numpy_array<nb::shape<-1>>(
            std::move(out), {static_cast<size_t>(n)}));
      },
      nb::arg("a"), nb::arg("b"));

  // ==== intersects ====
  m.def(
      name("intersects_pp_").c_str(),
      [](const PW &a, const PW &b) -> nb::object {
        if (!a.is_batch() && !b.is_batch()) {
          auto hit = dispatch_pair(
              [](const auto &pa, const auto &pb) -> bool {
                return tf::intersects(pa, pb);
              },
              a, b);
          return nb::cast(hit);
        }

        int n = batch_count_pair(a, b);
        tf::buffer<std::int8_t> out;
        out.allocate(n);
        auto *dst = out.data();

        auto compute = [&](int i) {
          dst[i] = dispatch_pair_at(
                       [](const auto &pa, const auto &pb) -> bool {
                         return tf::intersects(pa, pb);
                       },
                       a, i, b, i)
                       ? std::int8_t{1}
                       : std::int8_t{0};
        };

        if (n >= 1000)
          tf::parallel_for_each(tf::make_sequence_range(n), compute);
        else
          for (int i = 0; i < n; ++i)
            compute(i);

        return nb::cast(make_numpy_array<nb::shape<-1>>(
            std::move(out), {static_cast<size_t>(n)}));
      },
      nb::arg("a"), nb::arg("b"));

  // ==== ray_cast ====
  // config: optional pair of ndarrays (min_ts, max_ts). Python expands
  // scalar tuples and None to arrays before calling.
  using RayConfigArrays = std::pair<
      nb::ndarray<nb::numpy, RealT, nb::ndim<1>, nb::c_contig>,
      nb::ndarray<nb::numpy, RealT, nb::ndim<1>, nb::c_contig>>;
  m.def(
      name("ray_cast_pp_").c_str(),
      [](const PW &ray_pw, const PW &target,
         std::optional<RayConfigArrays> opt_config) -> nb::object {
        if (!ray_pw.is_batch() && !target.is_batch()) {
          // Single ray: config arrays have length 1 or absent
          auto config = tf::ray_config<RealT>{};
          if (opt_config)
            config = {opt_config->first.data()[0],
                      opt_config->second.data()[0]};
          auto ray = make_ray(ray_pw);
          auto result = dispatch_single(
              [&](const auto &t) { return tf::ray_cast(ray, t, config); },
              target);
          if (result)
            return nb::cast(static_cast<RealT>(result.t));
          return nb::none();
        }

        // Batch: Python always provides config arrays
        const RealT *min_ts = opt_config->first.data();
        const RealT *max_ts = opt_config->second.data();

        bool br = ray_pw.is_batch();
        bool bt = target.is_batch();
        int n = br ? ray_pw.count() : target.count();
        if (br && bt && ray_pw.count() != target.count())
          throw std::runtime_error("batch size mismatch");

        tf::buffer<RealT> ts;
        ts.allocate(n);
        auto *t_dst = ts.data();
        constexpr auto nan = std::numeric_limits<RealT>::quiet_NaN();

        auto compute = [&](int i) {
          auto ray_ptr = ray_pw.element_ptr(i);
          auto pts =
              tf::make_points<Dims>(tf::make_range(ray_ptr, 2 * Dims));
          auto ray = tf::make_ray_like(pts[0], pts[1].as_vector_view());

          dispatch_at(
              [&](const auto &t_view) {
                auto r = tf::ray_cast(
                    ray, t_view,
                    tf::ray_config<RealT>{min_ts[i], max_ts[i]});
                t_dst[i] = r ? static_cast<RealT>(r.t) : nan;
              },
              target, i);
        };

        if (n >= 1000)
          tf::parallel_for_each(tf::make_sequence_range(n), compute);
        else
          for (int i = 0; i < n; ++i)
            compute(i);

        return nb::cast(make_numpy_array<nb::shape<-1>>(
            std::move(ts), {static_cast<size_t>(n)}));
      },
      nb::arg("ray"), nb::arg("target"),
      nb::arg("config").none() = nb::none());

  // ==== closest_point_pair ====
  m.def(
      name("closest_point_pair_pp_").c_str(),
      [](const PW &a, const PW &b) -> nb::object {
        if (!a.is_batch() && !b.is_batch()) {
          auto r = dispatch_pair(
              [](const auto &pa,
                 const auto &pb) -> tf::metric_point_pair<RealT, Dims> {
                return tf::closest_metric_point_pair(pa, pb);
              },
              a, b);

          auto *pt0_data = new RealT[Dims];
          auto *pt1_data = new RealT[Dims];
          for (std::size_t j = 0; j < Dims; ++j) {
            pt0_data[j] = r.first[j];
            pt1_data[j] = r.second[j];
          }
          auto pt0_arr =
              make_numpy_array<nb::shape<Dims>>(pt0_data, {Dims});
          auto pt1_arr =
              make_numpy_array<nb::shape<Dims>>(pt1_data, {Dims});
          return nb::make_tuple(static_cast<RealT>(r.metric), pt0_arr,
                                pt1_arr);
        }

        int n = batch_count_pair(a, b);
        tf::buffer<RealT> dists;
        tf::buffer<RealT> pts0;
        tf::buffer<RealT> pts1;
        dists.allocate(n);
        pts0.allocate(static_cast<std::size_t>(n) * Dims);
        pts1.allocate(static_cast<std::size_t>(n) * Dims);
        auto *d_dst = dists.data();
        auto *p0_dst = pts0.data();
        auto *p1_dst = pts1.data();

        auto compute = [&](int i) {
          auto r = dispatch_pair_at(
              [](const auto &pa,
                 const auto &pb) -> tf::metric_point_pair<RealT, Dims> {
                return tf::closest_metric_point_pair(pa, pb);
              },
              a, i, b, i);
          d_dst[i] = static_cast<RealT>(r.metric);
          auto *p0 = p0_dst + static_cast<std::size_t>(i) * Dims;
          auto *p1 = p1_dst + static_cast<std::size_t>(i) * Dims;
          for (std::size_t j = 0; j < Dims; ++j) {
            p0[j] = r.first[j];
            p1[j] = r.second[j];
          }
        };

        if (n >= 1000)
          tf::parallel_for_each(tf::make_sequence_range(n), compute);
        else
          for (int i = 0; i < n; ++i)
            compute(i);

        auto d_arr = make_numpy_array<nb::shape<-1>>(
            std::move(dists), {static_cast<size_t>(n)});
        auto p0_arr = make_numpy_array<nb::shape<-1, Dims>>(
            std::move(pts0), {static_cast<size_t>(n), Dims});
        auto p1_arr = make_numpy_array<nb::shape<-1, Dims>>(
            std::move(pts1), {static_cast<size_t>(n), Dims});
        return nb::make_tuple(d_arr, p0_arr, p1_arr);
      },
      nb::arg("a"), nb::arg("b"));

  // ==== distance_field ====
  m.def(
      name("distance_field_pp_").c_str(),
      [](const PW &points, const PW &target) -> nb::object {
        int n = points.count();
        tf::buffer<RealT> out;
        out.allocate(n);
        auto *dst = out.data();

        dispatch_single(
            [&](const auto &t_view) {
              auto compute = [&](int i) {
                auto pt =
                    tf::make_point_view<Dims>(points.element_ptr(i));
                dst[i] = static_cast<RealT>(tf::distance(pt, t_view));
              };
              if (n >= 1000)
                tf::parallel_for_each(tf::make_sequence_range(n), compute);
              else
                for (int i = 0; i < n; ++i)
                  compute(i);
            },
            target);

        return nb::cast(make_numpy_array<nb::shape<-1>>(
            std::move(out), {static_cast<size_t>(n)}));
      },
      nb::arg("points"), nb::arg("target"));
}

} // anonymous namespace

auto register_prim_prim_ops(nanobind::module_ &m) -> void {
  register_prim_prim_ops_impl<2, float>(m, "float2d");
  register_prim_prim_ops_impl<3, float>(m, "float3d");
  register_prim_prim_ops_impl<2, double>(m, "double2d");
  register_prim_prim_ops_impl<3, double>(m, "double3d");
}

} // namespace tf::py
