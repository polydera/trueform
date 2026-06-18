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

#include "../../core/aabb.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_transform.hpp"
#include "../../core/buffer.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/inflated_aabb.hpp"
#include "../../core/intersects.hpp"
#include "../../core/local_value.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/take.hpp"
#include "../../exact/vertex.hpp"
#include "../../spatial/make_buffer_for_form.hpp"
#include "../../spatial/tree/dual_search.hpp"
#include "../../spatial/tree/self_search.hpp"
#include "../exact/face_plane_info.hpp"
#include "../exact/tagged_intersection.hpp"

namespace tf::intersect {

/// Per-thread scratch for the leaf-grouped face search. Converting + planing
/// each leaf face once here amortizes it over every candidate pair it shares in
/// the leaf; verts go in a flat arena keyed by per-face offsets (buffer<T>
/// can't nest), read back as `vertex_range`s. Also accumulates this thread's
/// output intersections + points.
template <typename Index, typename Int> struct face_pair_workspace {
  using vertex_t = tf::exact::vertex<Index, Int>;

  tf::buffer<tagged_intersection<Index>> intersections;
  tf::buffer<tf::exact::pt3<Int>> points;

  tf::buffer<vertex_t> verts0, verts1;
  tf::buffer<int> voff0, voff1;
  tf::buffer<tf::exact::face_plane<Int>> fp0, fp1;
  tf::buffer<tf::aabb<Int, 3>> ibox0, ibox1;
  tf::buffer<std::size_t> ids0, ids1;
  tf::buffer<bool> shared0, shared1; // self-intersection scratch (within path)

  void reset_cache() {
    verts0.clear();
    verts1.clear();
    voff0.clear();
    voff1.clear();
    ibox0.clear();
    ibox1.clear();
    ids0.clear();
    ids1.clear();
    voff0.push_back(0);
    voff1.push_back(0);
  }

  auto n0() const -> std::size_t { return ids0.size(); }
  auto n1() const -> std::size_t { return ids1.size(); }

  auto face0(std::size_t i) const -> tf::exact::vertex_range<Index, Int> {
    return tf::make_range(verts0.begin() + voff0[i], voff0[i + 1] - voff0[i]);
  }
  auto face1(std::size_t j) const -> tf::exact::vertex_range<Index, Int> {
    return tf::make_range(verts1.begin() + voff1[j], voff1[j + 1] - voff1[j]);
  }
};

/// Walk the dual BVH at leaf granularity, converting each leaf face once into
/// the per-thread workspace (side-1 aabb inflated by `pad`), then invoke
/// `process(ws)` per leaf pair to run the path-specific logic over the cache.
template <typename Form0, typename Form1, typename Conv, typename Index,
          typename Int, typename Process>
void search_face_pairs(const Form0 &form0, const Form1 &form1, int tag0,
                       int tag1, const Conv &conv, Int pad,
                       tf::local_value<face_pair_workspace<Index, Int>> &ws_lv,
                       const Process &process) {
  auto check_bvs = [&](const auto &bv0, const auto &bv1) {
    return tf::intersects(
        tf::make_aabb(conv.convert(bv0.min), conv.convert(bv0.max)),
        tf::inflated_aabb(
            tf::make_aabb(conv.convert(bv1.min), conv.convert(bv1.max)), pad));
  };
  auto bv_f = [&](const auto &bv0, const auto &bv1) -> bool {
    return check_bvs(tf::transformed(bv0, tf::frame_of(form0)),
                     tf::transformed(bv1, tf::frame_of(form1)));
  };
  auto ibox_of = [&](const auto &aabb, const auto &frame) {
    auto tb = tf::transformed(aabb, frame);
    return tf::make_aabb(conv.convert(tb.min), conv.convert(tb.max));
  };

  auto buff0 = tf::spatial::make_local_buffer_for_form(form0);
  auto buff1 = tf::spatial::make_local_buffer_for_form(form1);

  tf::spatial::impl::dual_search(
      form0.tree(), form1.tree(), bv_f,
      [&](const auto &r0, const auto &r1, const auto &aabbs0,
          const auto &aabbs1) -> bool {
        auto &ws = *ws_lv;
        ws.reset_cache();
        for (const auto &id0 : r0) {
          auto poly =
              tf::transformed(form0[id0] | tf::tag(buff0), tf::frame_of(form0));
          auto m = poly.size();
          for (decltype(m) k = 0; k < m; ++k)
            ws.verts0.push_back(conv(tag0, Index(poly.indices()[k]), poly[k]));
          ws.voff0.push_back(int(ws.verts0.size()));
          ws.ibox0.push_back(ibox_of(aabbs0[id0], tf::frame_of(form0)));
          ws.ids0.push_back(std::size_t(id0));
        }
        for (const auto &id1 : r1) {
          auto poly =
              tf::transformed(form1[id1] | tf::tag(buff1), tf::frame_of(form1));
          auto m = poly.size();
          for (decltype(m) k = 0; k < m; ++k)
            ws.verts1.push_back(conv(tag1, Index(poly.indices()[k]), poly[k]));
          ws.voff1.push_back(int(ws.verts1.size()));
          ws.ibox1.push_back(
              tf::inflated_aabb(ibox_of(aabbs1[id1], tf::frame_of(form1)), pad));
          ws.ids1.push_back(std::size_t(id1));
        }
        process(ws);
        return false;
      },
      [] { return false; }, 6);
}

/// Self-intersection variant: walk one form's BVH against itself. `impl
/// self_search` flags the diagonal leaf with `is_self`, where `process` must
/// start the inner loop at `i0 + 1` to skip the self-pair and each unordered
/// pair's mirror.
template <typename Form, typename Conv, typename Index, typename Int,
          typename Process>
void search_face_pairs_self(
    const Form &form, const Conv &conv, Int pad,
    tf::local_value<face_pair_workspace<Index, Int>> &ws_lv,
    const Process &process) {
  auto check_bvs = [&](const auto &bv0, const auto &bv1) {
    return tf::intersects(
        tf::make_aabb(conv.convert(bv0.min), conv.convert(bv0.max)),
        tf::inflated_aabb(
            tf::make_aabb(conv.convert(bv1.min), conv.convert(bv1.max)), pad));
  };
  auto bv_f = [&](const auto &bv0, const auto &bv1) -> bool {
    return check_bvs(tf::transformed(bv0, tf::frame_of(form)),
                     tf::transformed(bv1, tf::frame_of(form)));
  };
  auto ibox_of = [&](const auto &aabb) {
    auto tb = tf::transformed(aabb, tf::frame_of(form));
    return tf::make_aabb(conv.convert(tb.min), conv.convert(tb.max));
  };
  auto buff = tf::spatial::make_local_buffer_for_form(form);
  const auto &aabbs = form.tree().primitive_aabbs();

  tf::spatial::impl::self_search(
      form.tree(), bv_f,
      [&](const auto &r0, const auto &r1, bool is_self) -> bool {
        auto &ws = *ws_lv;
        ws.reset_cache();
        for (const auto &id0 : r0) {
          auto poly =
              tf::transformed(form[id0] | tf::tag(buff), tf::frame_of(form));
          auto m = poly.size();
          for (decltype(m) k = 0; k < m; ++k)
            ws.verts0.push_back(conv(0, Index(poly.indices()[k]), poly[k]));
          ws.voff0.push_back(int(ws.verts0.size()));
          ws.ibox0.push_back(ibox_of(aabbs[id0]));
          ws.ids0.push_back(std::size_t(id0));
        }
        for (const auto &id1 : r1) {
          auto poly =
              tf::transformed(form[id1] | tf::tag(buff), tf::frame_of(form));
          auto m = poly.size();
          for (decltype(m) k = 0; k < m; ++k)
            ws.verts1.push_back(conv(0, Index(poly.indices()[k]), poly[k]));
          ws.voff1.push_back(int(ws.verts1.size()));
          ws.ibox1.push_back(tf::inflated_aabb(ibox_of(aabbs[id1]), pad));
          ws.ids1.push_back(std::size_t(id1));
        }
        process(ws, is_self);
        return false;
      },
      [] { return false; }, 6);
}

/// Concatenate the per-thread workspaces; each intersection's point id is
/// rebased by its thread's point offset. Copies run per-thread in parallel.
template <typename Index, typename Int>
void merge_face_pair_workspaces(
    tf::local_value<face_pair_workspace<Index, Int>> &ws_lv,
    tf::buffer<tagged_intersection<Index>> &out_ints,
    tf::buffer<tf::exact::pt3<Int>> &out_points) {
  std::size_t n_pts = 0, n_ints = 0;
  for (const auto &ws : ws_lv.values()) {
    n_pts += ws.points.size();
    n_ints += ws.intersections.size();
  }
  out_points.allocate(n_pts);
  out_ints.allocate(n_ints);

  std::size_t poff = 0, ioff = 0;
  for (auto &ws : ws_lv.values()) {
    auto np = ws.points.size();
    auto ni = ws.intersections.size();
    tf::parallel_copy(ws.points, tf::take(tf::drop(out_points, poff), np));
    auto base = Index(poff);
    tf::parallel_transform(ws.intersections,
                           tf::take(tf::drop(out_ints, ioff), ni),
                           [base](tagged_intersection<Index> e) {
                             e.id += base;
                             return e;
                           });
    poff += np;
    ioff += ni;
  }
}

} // namespace tf::intersect
