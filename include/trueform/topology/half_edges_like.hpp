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

  auto half_edge(const half_edge_handle_t &he)
      -> tf::half_edge<index_type> & {
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

  auto face_half_edge_handles_buffer()
      -> tf::buffer<half_edge_handle_t> & {
    return this->face_half_edges_buffer();
  }

  auto face_half_edge_handles_buffer() const
      -> const tf::buffer<half_edge_handle_t> & {
    return this->face_half_edges_buffer();
  }

  auto vertex_half_edge_handles_buffer()
      -> tf::buffer<half_edge_handle_t> & {
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
    return tf::make_mapped_range(
        tf::make_sequence_range(index_type(hd.size())),
        [hd](index_type id) -> half_edge_handle_t {
          return hd[id].is_removed() ? half_edge_handle_t::invalid()
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

  auto half_edge_handle(const face_handle_t &fh) const
      -> half_edge_handle_t {
    return this->face_half_edges()[fh.id()];
  }

  auto half_edge_handle(const vertex_handle_t &vh) const
      -> half_edge_handle_t {
    return this->vertex_half_edges()[vh.id()];
  }

  auto edge_handle(const half_edge_handle_t &heh) const -> edge_handle_t {
    if (!heh.is_valid())
      return edge_handle_t::invalid();
    return edge_handle(tf::unsafe, heh);
  }

  // --- Handle conversions (unsafe) ---

  auto half_edge_handle(tf::unsafe_t, const edge_handle_t &eh,
                        bool side) const -> half_edge_handle_t {
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

  auto previous(half_edge_handle_t heh) const -> half_edge_handle_t {
    if (!heh.is_valid())
      return heh;
    if (is_boundary(tf::unsafe, heh)) {
      auto current_h = opposite(heh);
      auto next_h = next(current_h);
      if (!next_h.is_valid())
        return next_h;
      do {
        current_h = opposite(next_h);
        next_h = next(current_h);
        if (!next_h.is_valid())
          return next_h;
      } while (next_h != heh);
      return current_h;
    } else {
      auto current = heh;
      auto next_h = next(current);
      if (!next_h.is_valid())
        return next_h;
      while (next_h != heh) {
        current = next_h;
        next_h = next(next_h);
        if (!next_h.is_valid())
          return next_h;
      }
      return current;
    }
  }

  auto opposite(const half_edge_handle_t &he) const -> half_edge_handle_t {
    if (!he.is_valid())
      return he;
    return opposite(tf::unsafe, he);
  }

  auto rotated(const half_edge_handle_t &he) const -> half_edge_handle_t {
    return next(opposite(he));
  }

  auto anti_rotated(const half_edge_handle_t &he) const
      -> half_edge_handle_t {
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

  auto previous(tf::unsafe_t, half_edge_handle_t heh) const
      -> half_edge_handle_t {
    if (is_boundary(tf::unsafe, heh)) {
      auto current_h = opposite(tf::unsafe, heh);
      auto next_h = next(tf::unsafe, current_h);
      do {
        current_h = opposite(tf::unsafe, next_h);
        next_h = next(tf::unsafe, current_h);
      } while (next_h != heh);
      return current_h;
    } else {
      auto current = heh;
      auto next_h = next(tf::unsafe, current);
      while (next_h != heh) {
        current = next_h;
        next_h = next(tf::unsafe, next_h);
      }
      return current;
    }
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

  auto is_boundary(tf::unsafe_t, const half_edge_handle_t &he) const
      -> bool {
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

  auto is_simple(tf::unsafe_t, const half_edge_handle_t &he) const -> bool {
    return this->half_edges_data()[he.id()].is_simple();
  }

  auto is_simple(tf::unsafe_t, const edge_handle_t &eh) const -> bool {
    return is_simple(tf::unsafe, half_edge_handle(tf::unsafe, eh, 0)) &&
           is_simple(tf::unsafe, half_edge_handle(tf::unsafe, eh, 1));
  }

  auto is_manifold(tf::unsafe_t, const half_edge_handle_t &he) const
      -> bool {
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

    half_edge(b2).face = ha0.face;
    half_edge(a0).vertex = ha2.vertex;
    half_edge(a0).next = b2.id();
    half_edge(b2).next = a1.id();
    half_edge(a1).next = a0.id();

    this->face_half_edges()[hb2.face] = b0;
    this->vertex_half_edges()[ha0.vertex] = b1;

    half_edge(a2).face = hb0.face;
    half_edge(b0).vertex = hb2.vertex;
    half_edge(b0).next = a2.id();
    half_edge(a2).next = b1.id();
    half_edge(b1).next = b0.id();

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
    auto start_h = opposite(tf::unsafe, next(tf::unsafe, he0));
    if (!start_h.is_valid())
      return false;
    auto opp_next_he1 = opposite(tf::unsafe, next(tf::unsafe, he1));
    if (!opp_next_he1.is_valid())
      return false;
    auto v1 = half_edge(opp_next_he1).vertex;
    auto current = rotated(start_h);
    while (current.is_valid() && current != start_h) {
      auto tmp = opposite(tf::unsafe, current);
      if (!tmp.is_valid() || half_edge(tmp).vertex == v1)
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

  auto collapse(const half_edge_handle_t &heh_in) -> index_type {
    using Index = index_type;
    constexpr auto del = tf::half_edge<Index>::removed;
    auto heh_opp = opposite(tf::unsafe, heh_in);
    auto v0 = this->half_edges_data()[heh_in.id()].vertex;
    auto v1 = this->half_edges_data()[heh_opp.id()].vertex;

    bool has_fa = this->half_edges_data()[heh_in.id()].is_simple();
    bool has_fb = this->half_edges_data()[heh_opp.id()].is_simple();

    half_edge_handle_t a1, a2;
    Index face_a = -1;
    bool fa_degenerates = false;
    if (has_fa) {
      a1 = next(tf::unsafe, heh_in);
      a2 = previous(tf::unsafe, heh_in);
      face_a = this->half_edges_data()[heh_in.id()].face;
      fa_degenerates = (next(tf::unsafe, a1) == a2);
    }

    half_edge_handle_t b1, b2;
    Index face_b = -1;
    bool fb_degenerates = false;
    if (has_fb) {
      b1 = next(tf::unsafe, heh_opp);
      b2 = previous(tf::unsafe, heh_opp);
      face_b = this->half_edges_data()[heh_opp.id()].face;
      fb_degenerates = (next(tf::unsafe, b1) == b2);
    }

    half_edge_handle_t oa2, prev_oa2, ob1, prev_ob1;
    if (has_fa && fa_degenerates) {
      oa2 = opposite(tf::unsafe, a2);
      prev_oa2 = previous(tf::unsafe, oa2);
    }
    if (has_fb && fb_degenerates) {
      ob1 = opposite(tf::unsafe, b1);
      prev_ob1 = previous(tf::unsafe, ob1);
    }

    half_edge_handle_t prev_bnd;
    if (!has_fa)
      prev_bnd = previous(tf::unsafe, heh_in);
    if (!has_fb)
      prev_bnd = previous(tf::unsafe, heh_opp);

    auto ring_he = heh_opp;
    do {
      this->half_edges_data()[ring_he.id()].vertex = v0;
      ring_he = rotated(tf::unsafe, ring_he);
    } while (ring_he != heh_opp);

    Index faces_removed = 0;
    if (has_fa && fa_degenerates) {
      this->half_edges_data()[a1.id()] = this->half_edges_data()[oa2.id()];
      this->half_edges_data()[prev_oa2.id()].next = a1.id();

      auto face_d = this->half_edges_data()[a1.id()].face;
      if (face_d >= 0)
        this->face_half_edges()[face_d] = half_edge_handle_t{a1.id()};

      this->half_edges_data()[a2.id()].face = del;
      this->half_edges_data()[oa2.id()].face = del;
      this->face_half_edges()[face_a] = half_edge_handle_t::invalid();
      ++faces_removed;
    } else if (has_fa) {
      this->half_edges_data()[a2.id()].next = a1.id();
      this->face_half_edges()[face_a] = a1;
    }

    if (has_fb && fb_degenerates) {
      if (has_fa && fa_degenerates && prev_ob1 == oa2)
        prev_ob1 = a1;

      this->half_edges_data()[b2.id()] = this->half_edges_data()[ob1.id()];
      this->half_edges_data()[prev_ob1.id()].next = b2.id();

      auto face_e = this->half_edges_data()[b2.id()].face;
      if (face_e >= 0)
        this->face_half_edges()[face_e] = half_edge_handle_t{b2.id()};

      this->half_edges_data()[b1.id()].face = del;
      this->half_edges_data()[ob1.id()].face = del;
      this->face_half_edges()[face_b] = half_edge_handle_t::invalid();
      ++faces_removed;
    } else if (has_fb) {
      this->half_edges_data()[b2.id()].next = b1.id();
      this->face_half_edges()[face_b] = b1;
    }

    if (!has_fa)
      this->half_edges_data()[prev_bnd.id()].next =
          this->half_edges_data()[heh_in.id()].next;
    if (!has_fb)
      this->half_edges_data()[prev_bnd.id()].next =
          this->half_edges_data()[heh_opp.id()].next;

    this->half_edges_data()[heh_in.id()].face = del;
    this->half_edges_data()[heh_opp.id()].face = del;

    this->vertex_half_edges()[v1] = half_edge_handle_t::invalid();

    if (has_fa && fa_degenerates) {
      auto oa1 = opposite(tf::unsafe, a1);
      this->vertex_half_edges()[v0] = half_edge_handle_t{a1.id()};
      auto v2 = this->half_edges_data()[oa1.id()].vertex;
      if (this->vertex_half_edges()[v2].id() == a2.id())
        this->vertex_half_edges()[v2] = oa1;
    } else if (has_fa) {
      this->vertex_half_edges()[v0] = a1;
    }
    if (has_fb && fb_degenerates) {
      auto ob2 = opposite(tf::unsafe, b2);
      if (!has_fa || !fa_degenerates)
        this->vertex_half_edges()[v0] = ob2;
      auto v3 = this->half_edges_data()[b2.id()].vertex;
      if (this->vertex_half_edges()[v3].id() == ob1.id())
        this->vertex_half_edges()[v3] = half_edge_handle_t{b2.id()};
    } else if (has_fb) {
      if (!has_fa)
        this->vertex_half_edges()[v0] = b1;
    }

    this->_n_faces -= faces_removed;
    --this->_n_vertices;

    return faces_removed;
  }

  auto collapse(const edge_handle_t &eh) -> index_type {
    return collapse(half_edge_handle(tf::unsafe, eh, false));
  }

private:
  template <typename Ring>
  auto is_collapse_ok_impl(const edge_handle_t &eh, Ring &ring) const
      -> bool {
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
    return shared <= 2;
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
      he.half_edges_data(), he.face_half_edges(), he.vertex_half_edges(),
      he.boundary_vertex_data(), he.n_faces(), he.n_vertices()};
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
