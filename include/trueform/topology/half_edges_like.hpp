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

#include <algorithm>
#include <utility>
#include <vector>

#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/blocked_buffer.hpp"
#include "../core/buffer.hpp"
#include "../core/small_vector.hpp"
#include "../core/unsafe.hpp"
#include "../core/views/mapped_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "./half_edge.hpp"
#include "./structures/half_edges_ranges.hpp"

namespace tf {

template <typename Policy> class half_edges_like : public Policy {
public:
  half_edges_like() = default;
  half_edges_like(const Policy &p) : Policy{p} {}
  half_edges_like(Policy &&p) : Policy{std::move(p)} {}

  using Policy::Policy;

  using Policy::boundary_vertex_data;
  using Policy::face_half_edges;
  using Policy::half_edges_data;
  using Policy::n_faces;
  using Policy::n_vertices;
  using Policy::non_manifold_vertex_data;
  using Policy::vertex_half_edges;

  using typename Policy::edge_handle_t;
  using typename Policy::face_handle_t;
  using typename Policy::half_edge_handle_t;
  using typename Policy::index_type;
  using typename Policy::vertex_handle_t;

  auto half_edge(const half_edge_handle_t &he) const
      -> const tf::half_edge<index_type> & {
    return this->half_edges_data()[he.id()];
  }

  auto half_edge(const half_edge_handle_t &he) -> tf::half_edge<index_type> & {
    return this->half_edges_data()[he.id()];
  }

  auto half_edges_buffer() const
      -> const tf::buffer<tf::half_edge<index_type>> & {
    return this->half_edges_data_buffer();
  }

  auto half_edges_buffer() -> tf::buffer<tf::half_edge<index_type>> & {
    return this->half_edges_data_buffer();
  }

  auto face_half_edge_handles() const { return this->face_half_edges(); }

  auto vertex_half_edge_handles() const { return this->vertex_half_edges(); }

  auto face_half_edge_handles_buffer() -> tf::buffer<half_edge_handle_t> & {
    return this->face_half_edges_buffer();
  }

  auto face_half_edge_handles_buffer() const
      -> const tf::buffer<half_edge_handle_t> & {
    return this->face_half_edges_buffer();
  }

  auto vertex_half_edge_handles_buffer() -> tf::buffer<half_edge_handle_t> & {
    return this->vertex_half_edges_buffer();
  }

  auto vertex_half_edge_handles_buffer() const
      -> const tf::buffer<half_edge_handle_t> & {
    return this->vertex_half_edges_buffer();
  }

  auto boundary_vertices() const -> const tf::buffer<char> & {
    return this->boundary_vertex_data_buffer();
  }

  auto number_of_faces() const -> index_type { return this->n_faces(); }

  auto number_of_vertices() const -> index_type { return this->n_vertices(); }

  // --- Handle ranges ---

  auto edge_handles() const {
    auto hd = this->half_edges_data();
    return tf::make_mapped_range(
        tf::make_sequence_range(index_type(hd.size() / 2)),
        [hd](index_type id) -> edge_handle_t {
          return hd[id << 1].is_removed() ? edge_handle_t::invalid()
                                          : edge_handle_t{id};
        });
  }

  auto half_edge_handles() const {
    auto hd = this->half_edges_data();
    return tf::make_mapped_range(tf::make_sequence_range(index_type(hd.size())),
                                 [hd](index_type id) -> half_edge_handle_t {
                                   return hd[id].is_removed()
                                              ? half_edge_handle_t::invalid()
                                              : half_edge_handle_t{id};
                                 });
  }

  // --- Counts ---

  auto recount() -> void {
    this->_n_faces = 0;
    for (auto fhe : this->face_half_edges())
      if (fhe.is_valid())
        ++this->_n_faces;
    this->_n_vertices = 0;
    for (auto vhe : this->vertex_half_edges())
      if (vhe.is_valid())
        ++this->_n_vertices;
  }

  // --- Handle conversions (safe) ---

  auto half_edge_handle(const edge_handle_t &eh, bool side) const
      -> half_edge_handle_t {
    if (!eh.is_valid())
      return half_edge_handle_t::invalid();
    return half_edge_handle(tf::unsafe, eh, side);
  }

  auto half_edge_handle(const face_handle_t &fh) const -> half_edge_handle_t {
    return this->face_half_edges()[fh.id()];
  }

  auto half_edge_handle(const vertex_handle_t &vh) const -> half_edge_handle_t {
    return this->vertex_half_edges()[vh.id()];
  }

  auto edge_handle(const half_edge_handle_t &heh) const -> edge_handle_t {
    if (!heh.is_valid())
      return edge_handle_t::invalid();
    return edge_handle(tf::unsafe, heh);
  }

  // --- Handle conversions (unsafe) ---

  auto half_edge_handle(tf::unsafe_t, const edge_handle_t &eh, bool side) const
      -> half_edge_handle_t {
    return half_edge_handle_t((eh.id() << 1) + side);
  }

  auto edge_handle(tf::unsafe_t, const half_edge_handle_t &heh) const
      -> edge_handle_t {
    return edge_handle_t(heh.id() >> 1);
  }

  // --- Navigation (safe) ---

  auto start_vertex_handle(const half_edge_handle_t &heh) const
      -> vertex_handle_t {
    if (!heh.is_valid())
      return vertex_handle_t::invalid();
    return start_vertex_handle(tf::unsafe, heh);
  }

  auto end_vertex_handle(const half_edge_handle_t &heh) const
      -> vertex_handle_t {
    if (!heh.is_valid())
      return vertex_handle_t::invalid();
    return start_vertex_handle(opposite(heh));
  }

  auto face_handle(const half_edge_handle_t &heh) const -> face_handle_t {
    if (!heh.is_valid())
      return face_handle_t::invalid();
    return face_handle(tf::unsafe, heh);
  }

  auto next(const half_edge_handle_t &he) const -> half_edge_handle_t {
    if (!he.is_valid())
      return he;
    if (!half_edge(he).is_manifold())
      return half_edge_handle_t::invalid();
    return next(tf::unsafe, he);
  }

  auto previous(const half_edge_handle_t &heh) const -> half_edge_handle_t {
    if (!heh.is_valid())
      return heh;
    if (!half_edge(heh).is_manifold())
      return half_edge_handle_t::invalid();
    return previous(tf::unsafe, heh);
  }

  auto opposite(const half_edge_handle_t &he) const -> half_edge_handle_t {
    if (!he.is_valid())
      return he;
    return opposite(tf::unsafe, he);
  }

  auto rotated(const half_edge_handle_t &he) const -> half_edge_handle_t {
    return next(opposite(he));
  }

  auto anti_rotated(const half_edge_handle_t &he) const -> half_edge_handle_t {
    return opposite(previous(he));
  }

  // --- Navigation (unsafe) ---

  auto start_vertex_handle(tf::unsafe_t, const half_edge_handle_t &heh) const
      -> vertex_handle_t {
    return vertex_handle_t{this->half_edges_data()[heh.id()].vertex};
  }

  auto end_vertex_handle(tf::unsafe_t, const half_edge_handle_t &heh) const
      -> vertex_handle_t {
    return start_vertex_handle(tf::unsafe, opposite(tf::unsafe, heh));
  }

  auto face_handle(tf::unsafe_t, const half_edge_handle_t &heh) const
      -> face_handle_t {
    return face_handle_t{this->half_edges_data()[heh.id()].face};
  }

  auto next(tf::unsafe_t, const half_edge_handle_t &he) const
      -> half_edge_handle_t {
    return half_edge_handle_t(half_edge(he).next);
  }

  auto previous(tf::unsafe_t, const half_edge_handle_t &heh) const
      -> half_edge_handle_t {
    return half_edge_handle_t(half_edge(heh).prev);
  }

  auto opposite(tf::unsafe_t, const half_edge_handle_t &he) const
      -> half_edge_handle_t {
    return half_edge_handle_t(he.id() ^ 1);
  }

  auto rotated(tf::unsafe_t, const half_edge_handle_t &he) const
      -> half_edge_handle_t {
    return next(tf::unsafe, opposite(tf::unsafe, he));
  }

  auto anti_rotated(tf::unsafe_t, const half_edge_handle_t &he) const
      -> half_edge_handle_t {
    return opposite(tf::unsafe, previous(tf::unsafe, he));
  }

  // --- Queries (safe) ---

  auto is_boundary(const half_edge_handle_t &he) const -> bool {
    if (!he.is_valid())
      return false;
    return is_boundary(tf::unsafe, he);
  }

  auto is_boundary(const edge_handle_t &eh) const -> bool {
    if (!eh.is_valid())
      return false;
    return is_boundary(tf::unsafe, eh);
  }

  auto is_simple(const half_edge_handle_t &he) const -> bool {
    if (!he.is_valid())
      return false;
    return is_simple(tf::unsafe, he);
  }

  auto is_simple(const edge_handle_t &eh) const -> bool {
    if (!eh.is_valid())
      return false;
    return is_simple(tf::unsafe, eh);
  }

  auto is_manifold(const half_edge_handle_t &he) const -> bool {
    if (!he.is_valid())
      return false;
    return is_manifold(tf::unsafe, he);
  }

  auto is_manifold(const edge_handle_t &eh) const -> bool {
    if (!eh.is_valid())
      return false;
    return is_manifold(tf::unsafe, eh);
  }

  // --- Queries (unsafe) ---

  auto is_boundary(tf::unsafe_t, const half_edge_handle_t &he) const -> bool {
    return this->half_edges_data()[he.id()].is_boundary();
  }

  auto is_boundary(tf::unsafe_t, const edge_handle_t &eh) const -> bool {
    return is_boundary(tf::unsafe, half_edge_handle(tf::unsafe, eh, 0)) ||
           is_boundary(tf::unsafe, half_edge_handle(tf::unsafe, eh, 1));
  }

  auto is_boundary_vertex(tf::unsafe_t, index_type v) const -> bool {
    return this->boundary_vertex_data()[v];
  }

  auto is_boundary_vertex(index_type v) const -> bool {
    return this->boundary_vertex_data()[v];
  }

  auto is_non_manifold_vertex(tf::unsafe_t, index_type v) const -> bool {
    return this->non_manifold_vertex_data()[v];
  }

  auto is_non_manifold_vertex(index_type v) const -> bool {
    return this->non_manifold_vertex_data()[v];
  }

  auto is_simple(tf::unsafe_t, const half_edge_handle_t &he) const -> bool {
    return this->half_edges_data()[he.id()].is_simple();
  }

  auto is_simple(tf::unsafe_t, const edge_handle_t &eh) const -> bool {
    return is_simple(tf::unsafe, half_edge_handle(tf::unsafe, eh, 0)) &&
           is_simple(tf::unsafe, half_edge_handle(tf::unsafe, eh, 1));
  }

  auto is_manifold(tf::unsafe_t, const half_edge_handle_t &he) const -> bool {
    return this->half_edges_data()[he.id()].is_manifold();
  }

  auto is_manifold(tf::unsafe_t, const edge_handle_t &eh) const -> bool {
    return is_manifold(tf::unsafe, half_edge_handle(tf::unsafe, eh, 0)) &&
           is_manifold(tf::unsafe, half_edge_handle(tf::unsafe, eh, 1));
  }

  // --- Operations ---

  auto flip(const edge_handle_t &eh) -> void {
    auto a0 = half_edge_handle(tf::unsafe, eh, 0);
    auto a1 = next(tf::unsafe, a0);
    auto a2 = next(tf::unsafe, a1);
    auto b0 = half_edge_handle(tf::unsafe, eh, 1);
    auto b1 = next(tf::unsafe, b0);
    auto b2 = next(tf::unsafe, b1);
    auto ha0 = half_edge(a0);
    auto ha2 = half_edge(a2);
    auto hb0 = half_edge(b0);
    auto hb2 = half_edge(b2);

    // New face A loop: a0 -> b2 -> a1 -> a0
    half_edge(b2).face = ha0.face;
    half_edge(a0).vertex = ha2.vertex;
    half_edge(a0).next = b2.id();
    half_edge(b2).prev = a0.id();
    half_edge(b2).next = a1.id();
    half_edge(a1).prev = b2.id();
    half_edge(a1).next = a0.id();
    half_edge(a0).prev = a1.id();

    this->face_half_edges()[hb2.face] = b0;
    this->vertex_half_edges()[ha0.vertex] = b1;

    // New face B loop: b0 -> a2 -> b1 -> b0
    half_edge(a2).face = hb0.face;
    half_edge(b0).vertex = hb2.vertex;
    half_edge(b0).next = a2.id();
    half_edge(a2).prev = b0.id();
    half_edge(a2).next = b1.id();
    half_edge(b1).prev = a2.id();
    half_edge(b1).next = b0.id();
    half_edge(b0).prev = b1.id();

    this->face_half_edges()[ha2.face] = a0;
    this->vertex_half_edges()[hb0.vertex] = a1;
  }

  auto is_flip_ok(const edge_handle_t &eh) const -> bool {
    if (!eh.is_valid())
      return false;
    auto he0 = half_edge_handle(tf::unsafe, eh, 0);
    auto he1 = half_edge_handle(tf::unsafe, eh, 1);
    if (is_boundary(tf::unsafe, he0) || is_boundary(tf::unsafe, he1))
      return false;
    if (!is_manifold(tf::unsafe, he0) || !is_manifold(tf::unsafe, he1))
      return false;
    auto v0 = start_vertex_handle(tf::unsafe, he0).id();
    auto v1 = start_vertex_handle(tf::unsafe, he1).id();
    auto apex_0 = end_vertex_handle(tf::unsafe, next(tf::unsafe, he0)).id();
    auto apex_1 = end_vertex_handle(tf::unsafe, next(tf::unsafe, he1)).id();
    if (is_non_manifold_vertex(tf::unsafe, v0) ||
        is_non_manifold_vertex(tf::unsafe, v1) ||
        is_non_manifold_vertex(tf::unsafe, apex_0) ||
        is_non_manifold_vertex(tf::unsafe, apex_1))
      return false;
    auto start_h = opposite(tf::unsafe, next(tf::unsafe, he0));
    if (!start_h.is_valid())
      return false;
    auto current = rotated(start_h);
    while (current.is_valid() && current != start_h) {
      auto tmp = opposite(tf::unsafe, current);
      if (!tmp.is_valid() || half_edge(tmp).vertex == apex_1)
        return false;
      current = rotated(current);
    }
    return true;
  }

  auto is_collapse_ok(const edge_handle_t &eh,
                      tf::buffer<index_type> &ring) const -> bool {
    return is_collapse_ok_impl(eh, ring);
  }

  template <unsigned N>
  auto is_collapse_ok(const edge_handle_t &eh,
                      tf::small_vector<index_type, N> &ring) const -> bool {
    return is_collapse_ok_impl(eh, ring);
  }

  auto is_collapse_ok(const edge_handle_t &eh,
                      std::vector<index_type> &ring) const -> bool {
    return is_collapse_ok_impl(eh, ring);
  }

  /// Collapse a half-edge, merging its end vertex into its start vertex.
  /// Direction: h0 : v0 -> v1. v0 = h0.origin survives, v1 is removed.
  ///
  /// For each side that is a real triangle, the two non-collapsed edges of
  /// that triangle become duplicates after the merge — they both connect v0
  /// and the apex in opposite directions. We handle this by physically
  /// copying one of the outside partners into the slot of the inside edge,
  /// preserving the `^1` opposite invariant: slot `n0 ^ 1 == o_n0` stays
  /// intact, and slot `n0` now holds `o_n1`'s data. After the move, the
  /// surviving pair `(slot n0, slot o_n0)` is a valid opposite pair for the
  /// merged edge.
  ///
  /// For each side that is on the boundary (no triangle), h0 is just spliced
  /// out of the boundary loop.
  auto collapse(const half_edge_handle_t &heh_in) -> index_type {
    using Index = index_type;
    constexpr auto del = tf::half_edge<Index>::removed;
    auto &&hes = this->half_edges_data();

    // --- Cache handles on both sides of the collapsing edge.
    auto h0 = heh_in;                   // v0 -> v1; v0 survives
    auto h1 = opposite(tf::unsafe, h0); // v1 -> v0

    auto f0 = hes[h0.id()].face;   // triangle left of h0, or boundary sentinel
    auto f1 = hes[h1.id()].face;   // triangle left of h1, or boundary sentinel
    auto v0 = hes[h0.id()].vertex; // surviving vertex
    auto v1 = hes[h1.id()].vertex; // removed vertex

    Index faces_removed = 0;
    // Will point to a surviving half-edge at v0 by the end.
    Index v0_new_rep = h0.id();

    // --- f0 side.
    if (f0 >= 0) {
      // Real triangle f0 = v0 -> v1 -> apex_0 -> v0, with h0 = v0 -> v1.
      auto n0 = next(tf::unsafe, h0);       // v1 -> apex_0
      auto n1 = previous(tf::unsafe, h0);   // apex_0 -> v0
      auto o_n1 = opposite(tf::unsafe, n1); // v0 -> apex_0 in neighbor face

      // Capture o_n1's face-loop neighbors BEFORE the move — we read from
      // o_n1's slot here, and the slot will still hold the old data when
      // we write to new_id below. For a boundary half-edge at an open end
      // of a non-closed Eulerian path, prev or next may be invalid.
      auto o_n1_prev = previous(tf::unsafe, o_n1);
      auto o_n1_next = next(tf::unsafe, o_n1);

      // Copy o_n1's data into n0's slot. After this, slot new_id is the
      // new physical location of o_n1. Slot new_id ^ 1 == o_n0 is
      // unchanged, so the opposite pair is (new o_n1, o_n0) — exactly the
      // two sides of the merged edge.
      auto new_id = n0.id();
      hes[new_id] = hes[o_n1.id()];

      // Rewire o_n1's old loop neighbors to the new location — skip sides
      // that don't exist (open boundary chain end).
      if (o_n1_prev.is_valid())
        hes[o_n1_prev.id()].next = new_id;
      if (o_n1_next.is_valid())
        hes[o_n1_next.id()].prev = new_id;

      // If the neighbor face's representative was the old o_n1 slot, point
      // it at the new location.
      auto neighbor_face = hes[new_id].face;
      if (neighbor_face >= 0 && this->face_half_edges()[neighbor_face] == o_n1)
        this->face_half_edges()[neighbor_face] = half_edge_handle_t(new_id);

      // Dead slots: h0 (collapsed edge on this side), n1 (in f0), and the
      // original o_n1 slot (replaced by the moved data at new_id).
      hes[h0.id()].face = del;
      hes[n1.id()].face = del;
      hes[o_n1.id()].face = del;

      this->face_half_edges()[f0] = half_edge_handle_t::invalid();
      ++faces_removed;

      v0_new_rep = new_id;
    } else {
      // Boundary side: splice h0 out of the boundary loop.
      auto hp = previous(tf::unsafe, h0);
      auto hn = next(tf::unsafe, h0);
      hes[hp.id()].next = hn.id();
      hes[hn.id()].prev = hp.id();
      hes[h0.id()].face = del;
    }

    // --- f1 side (symmetric).
    if (f1 >= 0) {
      // Real triangle f1 = v1 -> v0 -> apex_1 -> v1, with h1 = v1 -> v0.
      auto m0 = next(tf::unsafe, h1);       // v0 -> apex_1
      auto m1 = previous(tf::unsafe, h1);   // apex_1 -> v1
      auto o_m1 = opposite(tf::unsafe, m1); // v1 -> apex_1 in neighbor face

      auto o_m1_prev = previous(tf::unsafe, o_m1);
      auto o_m1_next = next(tf::unsafe, o_m1);

      auto new_id = m0.id();
      hes[new_id] = hes[o_m1.id()];

      if (o_m1_prev.is_valid())
        hes[o_m1_prev.id()].next = new_id;
      if (o_m1_next.is_valid())
        hes[o_m1_next.id()].prev = new_id;

      // If the neighbor face's representative was the old o_m1 slot, point
      // it at the new location.
      auto neighbor_face = hes[new_id].face;
      if (neighbor_face >= 0 && this->face_half_edges()[neighbor_face] == o_m1)
        this->face_half_edges()[neighbor_face] = half_edge_handle_t(new_id);

      hes[h1.id()].face = del;
      hes[m1.id()].face = del;
      hes[o_m1.id()].face = del;

      this->face_half_edges()[f1] = half_edge_handle_t::invalid();
      ++faces_removed;

      v0_new_rep = new_id;
    } else {
      // Boundary side: splice h1 out of the boundary loop.
      auto hp = previous(tf::unsafe, h1);
      auto hn = next(tf::unsafe, h1);
      hes[hp.id()].next = hn.id();
      hes[hn.id()].prev = hp.id();
      hes[h1.id()].face = del;
    }

    // Update vertex representatives and propagate v0 to every half-edge in
    // the merged fan that was originally at v1.
    this->vertex_half_edges()[v0] = half_edge_handle_t(v0_new_rep);
    this->vertex_half_edges()[v1] = half_edge_handle_t::invalid();
    adjust_half_edge_ring_vertex(v0);

    this->_n_faces -= faces_removed;
    --this->_n_vertices;
    return faces_removed;
  }

private:
  /// Walk v's outgoing fan via rotated and set .vertex = v on every visited
  /// half-edge. Called post-collapse so that half-edges coming from the
  /// collapsed vertex adopt the surviving vertex's id.
  auto adjust_half_edge_ring_vertex(index_type v) -> void {
    auto start = this->vertex_half_edges()[v];
    if (!start.is_valid())
      return;
    auto cur = start;
    do {
      this->half_edges_data()[cur.id()].vertex = v;
      cur = rotated(cur);
      if (!cur.is_valid())
        return;
    } while (cur != start);
  }

public:
  auto collapse(const edge_handle_t &eh) -> index_type {
    return collapse(half_edge_handle(tf::unsafe, eh, false));
  }

private:
  template <typename Ring>
  auto is_collapse_ok_impl(const edge_handle_t &eh, Ring &ring) const -> bool {
    using Index = index_type;
    ring.clear();
    if (!eh.is_valid())
      return false;
    if (!is_manifold(tf::unsafe, eh))
      return false;
    auto heh0 = half_edge_handle(tf::unsafe, eh, false);
    auto heh1 = opposite(tf::unsafe, heh0);
    auto v0 = start_vertex_handle(tf::unsafe, heh0).id();
    auto v1 = end_vertex_handle(tf::unsafe, heh0).id();
    // Non-manifold vertex (split fan): rotated() cannot cross between the
    // fan components, so adjust_half_edge_ring_vertex would leave stale
    // .vertex values in the unvisited components. Reject.
    if (is_non_manifold_vertex(tf::unsafe, v0) ||
        is_non_manifold_vertex(tf::unsafe, v1))
      return false;
    // Interior edge between two boundary vertices would merge two disjoint
    // boundary segments through an interior vertex — topology change, reject.
    if (is_boundary_vertex(v0) && is_boundary_vertex(v1) &&
        !is_boundary(tf::unsafe, eh))
      return false;
    // Doublet edge: the two faces incident to (v0, v1) share their third
    // vertex, so f0 and f1 are the same triangle with opposite orientation.
    // Our physical-move collapse aliases o_n1 with m0 here and ends up
    // marking a surviving slot as removed. Reject.
    if (!is_boundary(tf::unsafe, heh0) && !is_boundary(tf::unsafe, heh1)) {
      auto apex0 = end_vertex_handle(tf::unsafe, next(tf::unsafe, heh0)).id();
      auto apex1 = end_vertex_handle(tf::unsafe, next(tf::unsafe, heh1)).id();
      if (apex0 == apex1)
        return false;
    }
    auto start = rotated(heh0);
    if (!start.is_valid())
      return false;
    auto current = start;
    do {
      auto v = end_vertex_handle(tf::unsafe, current).id();
      if (v != v0 && v != v1)
        ring.push_back(v);
      current = rotated(current);
      if (!current.is_valid())
        return false;
    } while (current != start);
    start = rotated(heh1);
    if (!start.is_valid())
      return false;
    Index shared = 0;
    current = start;
    do {
      auto v = end_vertex_handle(tf::unsafe, current).id();
      if (v != v0 && v != v1) {
        if (std::find(ring.begin(), ring.end(), v) != ring.end())
          ++shared;
      }
      current = rotated(current);
      if (!current.is_valid())
        return false;
    } while (current != start);
    return shared <= (1 + !is_boundary(eh));
  }
};

template <typename Policy>
auto unwrap(const half_edges_like<Policy> &t) -> decltype(auto) {
  return static_cast<const Policy &>(t);
}

template <typename Policy>
auto unwrap(half_edges_like<Policy> &t) -> decltype(auto) {
  return static_cast<Policy &>(t);
}

template <typename Policy>
auto unwrap(half_edges_like<Policy> &&t) -> decltype(auto) {
  return static_cast<Policy &&>(t);
}

template <typename Policy, typename T>
auto wrap_like(const half_edges_like<Policy> &, T &&t) {
  return half_edges_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(half_edges_like<Policy> &, T &&t) {
  return half_edges_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(half_edges_like<Policy> &&, T &&t) {
  return half_edges_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Range> auto make_half_edges_like(Range &&r) {
  return half_edges_like<std::decay_t<Range>>{static_cast<Range &&>(r)};
}

template <typename Policy>
auto make_half_edges_view(const half_edges_like<Policy> &he) {
  using Index = typename Policy::index_type;
  return half_edges_like<topology::half_edges_ranges<Index>>{
      he.half_edges_data(),          he.face_half_edges(),
      he.vertex_half_edges(),        he.boundary_vertex_data(),
      he.non_manifold_vertex_data(), he.n_faces(),
      he.n_vertices()};
}

template <typename Policy>
auto make_faces_buffer(const half_edges_like<Policy> &he)
    -> tf::blocked_buffer<typename Policy::index_type, 3> {
  using Index = typename Policy::index_type;
  tf::blocked_buffer<Index, 3> faces;
  faces.allocate(he.n_faces());
  tf::parallel_for_each(he.face_half_edges(), [&](auto heh) {
    if (!heh.is_valid())
      return;
    auto f = he.face_handle(tf::unsafe, heh).id();
    auto h1 = he.next(tf::unsafe, heh);
    auto h2 = he.next(tf::unsafe, h1);
    faces[f][0] = he.start_vertex_handle(tf::unsafe, heh).id();
    faces[f][1] = he.start_vertex_handle(tf::unsafe, h1).id();
    faces[f][2] = he.start_vertex_handle(tf::unsafe, h2).id();
  });
  return faces;
}

} // namespace tf
