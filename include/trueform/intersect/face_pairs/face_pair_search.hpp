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
#include "../../core/small_vector.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/take.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/vertex.hpp"
#include "../../spatial/make_buffer_for_form.hpp"
#include "../../spatial/tree/dual_search.hpp"
#include "../../spatial/tree/self_search.hpp"
#include "../classify/face_plane_info.hpp"
#include "../records/tagged_intersection.hpp"

namespace tf::intersect {

/// Per-thread scratch for the leaf-grouped face search. Converting + planing
/// each leaf face once here amortizes it over every candidate pair it shares in
/// the leaf; verts go in a flat arena keyed by per-face offsets (buffer<T>
/// can't nest), read back as `vertex_range`s. Also accumulates this thread's
/// output intersections + payloads.
///
/// `Payload` is what one emission stores about its point: its lattice
/// position, or — for a caller that wants identity and no rounding —
/// @ref tf::exact::edge_fractions, the point's exact parameter on the
/// original edges it lies on. The kernels emit one slot per record
/// either way, so a record id, the sentinel base derived from the slot
/// count, and every id the emission order assigns are the same numbers
/// under both payloads.
template <typename Index, typename Int,
          typename Payload = tf::exact::pt3<Int>>
struct face_pair_workspace {
  using vertex_t = tf::exact::vertex<Index, Int>;
  using payload_t = Payload;

  tf::buffer<tagged_intersection<Index>> intersections;
  tf::buffer<Payload> payloads;
  // pair-level coplanarity facts: {tag0, obj0, tag1, obj1} appended by
  // the kernels when the banded masks read all-zero both ways —
  // independent of record emission (representative gating must not be
  // able to lose the fact)
  tf::buffer<std::array<Index, 4>> coplanar_pairs;

  tf::buffer<vertex_t> verts0, verts1;
  tf::buffer<int> voff0, voff1;
  tf::buffer<tf::exact::face_plane<Int>> fp0, fp1;
  tf::buffer<tf::aabb<Int, 3>> ibox0, ibox1;
  tf::buffer<std::size_t> ids0, ids1;
  tf::buffer<bool> shared0, shared1; // self-intersection scratch (within path)

  // The block's candidate pairs the box filter keeps, and the faces those
  // pairs reach. @ref tf::intersect::prepare_face_pair_block is the one
  // producer of both: the pair loops read the verdict instead of retesting
  // it, and only a reached face carries a plane in `fp0` / `fp1`.
  tf::buffer<bool> pair_kept;
  tf::buffer<bool> reached0, reached1;

  // Per-candidate-pair scratch: each face's vertices read against the
  // other face's plane, and the exact values behind those signs. A leaf
  // pair runs its kernel once per candidate pair it holds, so the
  // containers are carried and refilled here rather than built there.
  tf::small_vector<int, 16> signs0, signs1;
  tf::small_vector<typename tf::exact::meta<Int>::T2, 16> values0, values1;

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

/// The exact box of the corners a leaf face just contributed. The
/// corners are the geometry the kernels read, so this box holds the
/// face with nothing to spare — the descent pays for the door's motion,
/// the block filter does not.
template <typename Index, typename Int>
auto placed_face_box(const tf::buffer<tf::exact::vertex<Index, Int>> &verts,
                     std::size_t from) -> tf::aabb<Int, 3> {
  auto low = verts[from].pt;
  auto high = low;
  for (auto at = from + 1; at < verts.size(); ++at)
    for (std::size_t k = 0; k < 3; ++k) {
      if (verts[at].pt[k] < low[k])
        low[k] = verts[at].pt[k];
      if (verts[at].pt[k] > high[k])
        high[k] = verts[at].pt[k];
    }
  return tf::make_aabb(low, high);
}

/// Walk the dual BVH at leaf granularity, gathering each leaf face's
/// placed corners once into the per-thread workspace, then invoke
/// `process(ws)` per leaf pair to run the path-specific logic over the
/// cache.
///
/// The descent is where the door's motion is paid for, and nowhere else.
/// The trees index the caller's float faces and stay the authority; a
/// placed vertex stands at most `T` from the vertex whose box the tree
/// holds, so two placed faces that meet have boxes meeting after each is
/// grown by `T` — which is one box against the other grown by `2 T`.
/// Below the descent the corners themselves are in hand and the filter
/// is exact.
template <typename Form0, typename Form1, typename Lattice, typename Index,
          typename Int, typename Payload, typename Process>
void search_face_pairs(
    const Form0 &form0, const Form1 &form1, int tag0, int tag1,
    const Lattice &lattice,
    tf::local_value<face_pair_workspace<Index, Int, Payload>> &ws_lv,
    const Process &process) {
  const auto &conv = lattice.converter();
  const Int pad = Int(2) * lattice.tolerance_int();
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

  auto buff0 = tf::spatial::make_local_buffer_for_form(form0);
  auto buff1 = tf::spatial::make_local_buffer_for_form(form1);

  tf::spatial::traversal::dual_search(
      form0.tree(), form1.tree(), bv_f,
      [&](const auto &r0, const auto &r1, const auto &, const auto &) -> bool {
        auto &ws = *ws_lv;
        ws.reset_cache();
        for (const auto &id0 : r0) {
          auto poly =
              tf::transformed(form0[id0] | tf::tag(buff0), tf::frame_of(form0));
          auto m = poly.size();
          const auto from = ws.verts0.size();
          for (decltype(m) k = 0; k < m; ++k)
            ws.verts0.push_back(
                {lattice.flat_vertex(tag0, Index(poly.indices()[k])),
                 lattice.point(tag0, Index(poly.indices()[k]), poly[k])});
          ws.voff0.push_back(int(ws.verts0.size()));
          ws.ibox0.push_back(placed_face_box(ws.verts0, from));
          ws.ids0.push_back(std::size_t(id0));
        }
        for (const auto &id1 : r1) {
          auto poly =
              tf::transformed(form1[id1] | tf::tag(buff1), tf::frame_of(form1));
          auto m = poly.size();
          const auto from = ws.verts1.size();
          for (decltype(m) k = 0; k < m; ++k)
            ws.verts1.push_back(
                {lattice.flat_vertex(tag1, Index(poly.indices()[k])),
                 lattice.point(tag1, Index(poly.indices()[k]), poly[k])});
          ws.voff1.push_back(int(ws.verts1.size()));
          ws.ibox1.push_back(placed_face_box(ws.verts1, from));
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
///
/// A self record's vertex ids are the form's own, not the flat space's —
/// one form is all a self pair ever sees, and the identity tier rebases
/// them by the record's tag.
template <typename Form, typename Lattice, typename Index, typename Int,
          typename Payload, typename Process>
void search_face_pairs_self(
    const Form &form, int tag, const Lattice &lattice,
    tf::local_value<face_pair_workspace<Index, Int, Payload>> &ws_lv,
    const Process &process) {
  const auto &conv = lattice.converter();
  const Int pad = Int(2) * lattice.tolerance_int();
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
  auto buff = tf::spatial::make_local_buffer_for_form(form);

  tf::spatial::traversal::self_search(
      form.tree(), bv_f,
      [&](const auto &r0, const auto &r1, bool is_self) -> bool {
        auto &ws = *ws_lv;
        ws.reset_cache();
        for (const auto &id0 : r0) {
          auto poly =
              tf::transformed(form[id0] | tf::tag(buff), tf::frame_of(form));
          auto m = poly.size();
          const auto from = ws.verts0.size();
          for (decltype(m) k = 0; k < m; ++k)
            ws.verts0.push_back(
                {Index(poly.indices()[k]),
                 lattice.point(tag, Index(poly.indices()[k]), poly[k])});
          ws.voff0.push_back(int(ws.verts0.size()));
          ws.ibox0.push_back(placed_face_box(ws.verts0, from));
          ws.ids0.push_back(std::size_t(id0));
        }
        for (const auto &id1 : r1) {
          auto poly =
              tf::transformed(form[id1] | tf::tag(buff), tf::frame_of(form));
          auto m = poly.size();
          const auto from = ws.verts1.size();
          for (decltype(m) k = 0; k < m; ++k)
            ws.verts1.push_back(
                {Index(poly.indices()[k]),
                 lattice.point(tag, Index(poly.indices()[k]), poly[k])});
          ws.voff1.push_back(int(ws.verts1.size()));
          ws.ibox1.push_back(placed_face_box(ws.verts1, from));
          ws.ids1.push_back(std::size_t(id1));
        }
        process(ws, is_self);
        return false;
      },
      [] { return false; }, 6);
}

/// Concatenate the per-thread workspaces; each intersection's payload id is
/// rebased by its thread's payload offset. Copies run per-thread in parallel.
template <typename Index, typename Int, typename Payload>
void merge_face_pair_workspaces(
    tf::local_value<face_pair_workspace<Index, Int, Payload>> &ws_lv,
    tf::buffer<tagged_intersection<Index>> &out_ints,
    tf::buffer<Payload> &out_payloads,
    tf::buffer<std::array<Index, 4>> &out_coplanar_pairs) {
  std::size_t n_pts = 0, n_ints = 0, n_cps = 0;
  for (const auto &ws : ws_lv.values()) {
    n_pts += ws.payloads.size();
    n_ints += ws.intersections.size();
    n_cps += ws.coplanar_pairs.size();
  }
  out_payloads.allocate(n_pts);
  out_ints.allocate(n_ints);
  out_coplanar_pairs.allocate(n_cps);

  std::size_t poff = 0, ioff = 0, coff = 0;
  for (auto &ws : ws_lv.values()) {
    auto np = ws.payloads.size();
    auto ni = ws.intersections.size();
    auto nc = ws.coplanar_pairs.size();
    tf::parallel_copy(ws.payloads, tf::take(tf::drop(out_payloads, poff), np));
    tf::parallel_copy(ws.coplanar_pairs,
                      tf::take(tf::drop(out_coplanar_pairs, coff), nc));
    auto base = Index(poff);
    tf::parallel_transform(ws.intersections,
                           tf::take(tf::drop(out_ints, ioff), ni),
                           [base](tagged_intersection<Index> e) {
                             e.id += base;
                             return e;
                           });
    poff += np;
    ioff += ni;
    coff += nc;
  }
}

} // namespace tf::intersect
