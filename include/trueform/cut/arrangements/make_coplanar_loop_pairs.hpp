/*
 * Copyright (c) 2026 XLAB
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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_iota.hpp"
#include "../../core/buffer.hpp"
#include "../../core/none.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../face_cuts.hpp"
#include "../face_regions.hpp"

#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <type_traits>
#include <utility>

namespace tf::cut {

/// @ingroup cut
/// @brief One coincident loop pair: same vertex cycle, `opposing` iff
///        the windings differ. `loop_a < loop_b`.
template <typename Index> struct coplanar_loop_pair {
  Index loop_a, loop_b;
  bool opposing;
};

/// @ingroup cut
/// @brief Which loop pairs a coincidence scan may relate.
enum class coplanar_scope {
  between_forms, ///< skip same-tag pairs (cut_graph's between rule)
  within_form,   ///< skip same-object pairs (cut_graph's self rule)
  all            ///< no pairs skipped: complete cliques, sound fold maps
};

namespace impl {

/// Hole walks the caller's structure attaches to its loops (@ref
/// tf::face_regions): boundary coincidence alone is not identity there,
/// the hole sets must coincide too. Callers without holes pass
/// `tf::none` instead.
template <typename LoopHoles, typename Holes> struct hole_context {
  LoopHoles loop_holes;
  Holes holes;
};

template <typename LoopHoles, typename Holes>
auto make_hole_context(LoopHoles loop_holes, Holes holes)
    -> hole_context<LoopHoles, Holes> {
  return {loop_holes, holes};
}

/// Detect coincident loops by minimal-canonical-edge key: coincident
/// loops share every edge, hence the minimal one, so an argsort on the
/// (size, edge) key groups every stack; within an equal run the stored
/// anchor aligns both loops at the shared edge and one element-wise
/// walk is the exact verdict, its direction mismatch the winding flag.
/// `Scope` selects which pairs qualify (@ref coplanar_scope). A
/// non-simple loop (a hole patched into the boundary) can traverse its
/// minimal edge more than once; the stored anchor is only the fast
/// path, and on a mismatch the walk retries at every other occurrence
/// of the edge in the partner. With a @ref hole_context a matched
/// boundary additionally requires coincident hole sets; `tf::none`
/// skips the confirmation.
template <coplanar_scope Scope, typename Index, typename Structure,
          typename HoleContext>
auto make_coplanar_loop_pairs(const Structure &fc, const HoleContext &hole_ctx)
    -> tf::buffer<coplanar_loop_pair<Index>> {
  using vt = tf::intersect::graph::vertex<Index>;
  using source = tf::intersect::graph::vertex_source;

  static_assert(std::is_signed_v<Index>, "sentinels require signed Index");
  auto loops = fc.loops();
  auto descs = fc.descriptors();
  const Index n = static_cast<Index>(loops.size());
  tf::buffer<coplanar_loop_pair<Index>> pairs;
  if (n == 0)
    return pairs;

  // Vertex identity is (source, id); original ids are per-form, so the
  // key carries an effective tag (created -> -1) to disambiguate.
  struct vkey_t {
    Index tag, src, id;
    auto operator<(const vkey_t &o) const -> bool {
      if (tag != o.tag)
        return tag < o.tag;
      if (src != o.src)
        return src < o.src;
      return id < o.id;
    }
    auto operator==(const vkey_t &o) const -> bool {
      return tag == o.tag && src == o.src && id == o.id;
    }
    auto operator!=(const vkey_t &o) const -> bool { return !(*this == o); }
  };
  // Loop size is part of the key — different sizes cannot be
  // coincident, so equal runs are size-pure. The edge leads the
  // comparison: it discriminates, sizes mostly repeat.
  struct ekey_t {
    vkey_t u, v;
    Index size;
    auto operator<(const ekey_t &o) const -> bool {
      if (u < o.u)
        return true;
      if (o.u < u)
        return false;
      if (v < o.v)
        return true;
      if (o.v < v)
        return false;
      return size < o.size;
    }
    auto operator==(const ekey_t &o) const -> bool {
      return u == o.u && v == o.v && size == o.size;
    }
  };
  auto vkey = [](const vt &v, Index tag) -> vkey_t {
    const bool created = v.source == source::created;
    return {created ? Index(-1) : tag, static_cast<Index>(v.source), v.id};
  };

  // Anchor: where the minimal edge sits in the loop and whether the
  // loop traverses it in canonical (a < b) direction — alignment for
  // the verification walk is then free.
  tf::buffer<ekey_t> keys;
  keys.allocate(static_cast<std::size_t>(n));
  tf::buffer<std::array<Index, 2>> anchors; // (position, forward)
  anchors.allocate(static_cast<std::size_t>(n));
  tf::parallel_for_each(tf::make_sequence_range(n), [&](Index l) {
    const Index tag = static_cast<Index>(descs[l].tag);
    auto loop = loops[l];
    const Index m = static_cast<Index>(loop.size());
    ekey_t best{};
    Index best_pos = 0, best_fwd = 1;
    for (Index i = 0; i < m; ++i) {
      vkey_t a = vkey(loop[i], tag);
      vkey_t b = vkey(loop[i + 1 == m ? Index(0) : i + 1], tag);
      // zero-length edges are compacted away upstream, so a != b
      const bool fwd = a < b;
      if (!fwd)
        std::swap(a, b);
      const ekey_t e{a, b, m};
      if (i == 0 || e < best) {
        best = e;
        best_pos = i;
        best_fwd = Index(fwd);
      }
    }
    keys[l] = best;
    anchors[l] = {best_pos, best_fwd};
  });

  tf::buffer<Index> order;
  order.allocate(static_cast<std::size_t>(n));
  tf::parallel_iota(order, 0);
  tbb::parallel_sort(order.begin(), order.end(), [&](Index x, Index y) {
    if (keys[x] < keys[y])
      return true;
    if (keys[y] < keys[x])
      return false;
    return x < y;
  });

  // Hole confirmation: keys are winding-canonical, so a sorted multiset
  // of per-hole minimal-edge keys coincides exactly when the hole sets
  // do — under either boundary winding. Separate scratch from the
  // boundary keys/anchors arrays, which stay live for the pairing scan.
  tf::buffer<ekey_t> hole_keys_a, hole_keys_b;
  auto holes_coincident = [&](Index la, Index lb) -> bool {
    if constexpr (!std::is_same_v<HoleContext, tf::none_t>) {
      auto ha = hole_ctx.loop_holes[la];
      auto hb = hole_ctx.loop_holes[lb];
      if (ha.size() != hb.size())
        return false;
      if (ha.size() == 0)
        return true;
      auto hole_key = [&](const auto &hole, Index tag) -> ekey_t {
        const Index m = static_cast<Index>(hole.size());
        ekey_t best{};
        for (Index i = 0; i < m; ++i) {
          vkey_t a = vkey(hole[i], tag);
          vkey_t b = vkey(hole[i + 1 == m ? Index(0) : i + 1], tag);
          if (b < a)
            std::swap(a, b);
          const ekey_t e{a, b, m};
          if (i == 0 || e < best)
            best = e;
        }
        return best;
      };
      const Index ta = static_cast<Index>(descs[la].tag);
      const Index tb = static_cast<Index>(descs[lb].tag);
      auto hs = hole_ctx.holes;
      hole_keys_a.clear();
      hole_keys_b.clear();
      for (auto h : ha)
        hole_keys_a.push_back(hole_key(hs[h], ta));
      for (auto h : hb)
        hole_keys_b.push_back(hole_key(hs[h], tb));
      std::sort(hole_keys_a.begin(), hole_keys_a.end());
      std::sort(hole_keys_b.begin(), hole_keys_b.end());
      return std::equal(hole_keys_a.begin(), hole_keys_a.end(),
                        hole_keys_b.begin());
    } else {
      return true;
    }
  };

  // Anchored verification: equal keys share the minimal edge, so both
  // loops align at their stored positions — one element-wise walk,
  // forward against forward or forward against backward. A non-simple
  // loop can traverse the minimal edge more than once and the stored
  // anchors can then align the wrong copies, so a failed walk retries
  // at the partner's other occurrences of the edge.
  auto coincident = [&](Index la, Index lb) -> int {
    auto x = loops[la];
    auto y = loops[lb];
    const Index m = static_cast<Index>(x.size());
    const Index tx = static_cast<Index>(descs[la].tag);
    const Index ty = static_cast<Index>(descs[lb].tag);
    const Index px = anchors[la][0];
    const bool fx = bool(anchors[la][1]);
    auto walk = [&](Index py, bool opposing) -> bool {
      for (Index k = 0; k < m; ++k) {
        const Index ix = px + k >= m ? px + k - m : px + k;
        Index iy;
        if (!opposing)
          iy = py + k >= m ? py + k - m : py + k;
        else {
          const Index r = py + 1 - k;
          iy = r < 0 ? r + m : (r >= m ? r - m : r);
        }
        if (vkey(x[ix], tx) != vkey(y[iy], ty))
          return false;
      }
      return true;
    };
    const Index py0 = anchors[lb][0];
    const bool opp0 = fx != bool(anchors[lb][1]);
    if (walk(py0, opp0) && holes_coincident(la, lb))
      return opp0 ? -1 : 1;
    for (Index j = 0; j < m; ++j) {
      if (j == py0)
        continue;
      vkey_t a = vkey(y[j], ty);
      vkey_t b = vkey(y[j + 1 == m ? Index(0) : j + 1], ty);
      const bool fwd = a < b;
      if (!fwd)
        std::swap(a, b);
      if (a != keys[la].u || b != keys[la].v)
        continue;
      const bool opp = fx != fwd;
      if (walk(j, opp) && holes_coincident(la, lb))
        return opp ? -1 : 1;
    }
    return 0;
  };

  // Pairwise within each equal-key run: runs are coincident stacks
  // (shallow) plus rare same-min-edge fans whose walks abort at the
  // first off-edge vertex — same bound as cut_graph's neighbor scan.
  for (Index i = 0; i < n;) {
    Index j = i + 1;
    while (j < n && keys[order[i]] == keys[order[j]])
      ++j;
    for (Index p = i; p < j; ++p)
      for (Index q = p + 1; q < j; ++q) {
        const Index la = order[p] < order[q] ? order[p] : order[q];
        const Index lb = order[p] < order[q] ? order[q] : order[p];
        if constexpr (Scope == coplanar_scope::within_form) {
          if (descs[la].object == descs[lb].object)
            continue;
        } else if constexpr (Scope == coplanar_scope::between_forms) {
          if (descs[la].tag == descs[lb].tag)
            continue;
        }
        const int cmp = coincident(la, lb);
        if (cmp != 0)
          pairs.push_back({la, lb, cmp < 0});
      }
    i = j;
  }
  return pairs;
}

} // namespace impl

/// @ingroup cut
/// @brief Coincident loop pairs across forms (skips same-tag pairs).
///
/// The pairs of a k-deep coincident stack form its full clique; the
/// stack's smallest loop id never appears as `loop_b`.
template <typename Index, typename Int>
auto make_coplanar_loop_pairs(const tf::face_cuts<Index, Int> &fc)
    -> tf::buffer<coplanar_loop_pair<Index>> {
  return impl::make_coplanar_loop_pairs<coplanar_scope::between_forms, Index>(
      fc, tf::none);
}

/// @ingroup cut
/// @brief Coincident region pairs across forms; a matched boundary must
///        also carry a coincident hole set.
/// @overload
template <typename Index, typename Int>
auto make_coplanar_loop_pairs(const tf::face_regions<Index, Int> &fr)
    -> tf::buffer<coplanar_loop_pair<Index>> {
  return impl::make_coplanar_loop_pairs<coplanar_scope::between_forms, Index>(
      fr, impl::make_hole_context(fr.loop_holes(), fr.holes()));
}

/// @ingroup cut
/// @brief Coincident loop pairs within one form (skips same-face pairs).
/// @overload
template <typename Index, typename Int>
auto make_coplanar_loop_pairs_self(const tf::face_cuts<Index, Int> &fc)
    -> tf::buffer<coplanar_loop_pair<Index>> {
  return impl::make_coplanar_loop_pairs<coplanar_scope::within_form, Index>(
      fc, tf::none);
}

/// @ingroup cut
/// @brief Coincident region pairs within one form.
/// @overload
template <typename Index, typename Int>
auto make_coplanar_loop_pairs_self(const tf::face_regions<Index, Int> &fr)
    -> tf::buffer<coplanar_loop_pair<Index>> {
  return impl::make_coplanar_loop_pairs<coplanar_scope::within_form, Index>(
      fr, impl::make_hole_context(fr.loop_holes(), fr.holes()));
}

/// @ingroup cut
/// @brief Coincident loop pairs of any two distinct loops — the scope
///        for triangulation sharing, where a fold is only ever an
///        emission equality, never a dedup. Nothing is skipped, so a
///        stack's clique is complete and every dead loop pairs with
///        the live minimum.
/// @overload
template <typename Index, typename Int>
auto make_coplanar_loop_pairs_all(const tf::face_cuts<Index, Int> &fc)
    -> tf::buffer<coplanar_loop_pair<Index>> {
  return impl::make_coplanar_loop_pairs<coplanar_scope::all, Index>(fc,
                                                                    tf::none);
}

/// @ingroup cut
/// @brief Coincident region pairs of any two distinct regions.
/// @overload
template <typename Index, typename Int>
auto make_coplanar_loop_pairs_all(const tf::face_regions<Index, Int> &fr)
    -> tf::buffer<coplanar_loop_pair<Index>> {
  return impl::make_coplanar_loop_pairs<coplanar_scope::all, Index>(
      fr, impl::make_hole_context(fr.loop_holes(), fr.holes()));
}

/// @ingroup cut
/// @brief Per-loop fold lookup from coincident pairs.
///
/// `fold_of[l] = {survivor, reversed}` for every dead loop, folded onto
/// its stack's smallest LIVE loop id; `{-1, 0}` for live loops.
///
/// @pre Every dead loop should appear in at least one pair whose
///      `loop_a` is live — guaranteed for `_all` and `_self` scoped
///      pairs (the clique's minimum is live and pairs with everyone).
///      Between-forms pairs on self-coincident forms can violate it;
///      such an entry then folds onto a dead loop with the direct
///      pair's flag — internally consistent, chase `fold_of[survivor]`.
template <typename Index>
auto make_loop_fold_map(const tf::buffer<coplanar_loop_pair<Index>> &pairs,
                        std::size_t n_loops)
    -> tf::buffer<std::array<Index, 2>> {
  tf::buffer<std::array<Index, 2>> fold_of;
  fold_of.allocate(n_loops);
  tf::parallel_for_each(fold_of,
                        [](auto &f) { f = {Index(-1), Index(0)}; });
  // Pass 1 marks every loop_b dead (flag kept pair-consistent); pass 2
  // rebinds each dead loop to its smallest live survivor.
  for (const auto &p : pairs)
    fold_of[p.loop_b] = {p.loop_a, Index(p.opposing)};
  for (const auto &p : pairs)
    if (fold_of[p.loop_a][0] == Index(-1)) {
      auto &f = fold_of[p.loop_b];
      if (fold_of[f[0]][0] != Index(-1) || p.loop_a < f[0])
        f = {p.loop_a, Index(p.opposing)};
    }
  return fold_of;
}

} // namespace tf::cut
