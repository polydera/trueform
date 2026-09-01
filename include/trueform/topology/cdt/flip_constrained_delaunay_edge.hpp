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
#include <cstddef>
#include <initializer_list>

namespace tf::topology::cdt {

template <typename Owner>
auto flip_constrained_delaunay_edge(Owner &owner,
                                    typename Owner::index_type edge) -> void {
  using Index = typename Owner::index_type;
  const Index opposite = Owner::opposite(edge);
  const Index edge01 = owner.previous_edge(edge);
  const Index edge03 = owner.next_edge(edge);
  const Index edge21 = owner.next_edge(opposite);
  const Index edge23 = owner.previous_edge(opposite);

  owner._edges[std::size_t(edge01)].next = edge03;
  owner._edges[std::size_t(edge03)].prev = edge01;
  owner._edges[std::size_t(edge23)].next = edge21;
  owner._edges[std::size_t(edge21)].prev = edge23;

  owner._v_first_edge[std::size_t(owner.origin(edge))] = edge01;
  owner._v_first_edge[std::size_t(owner.origin(opposite))] = edge23;

  const Index edge10 = Owner::opposite(edge01);
  const Index edge12 = Owner::opposite(edge21);
  const Index edge30 = Owner::opposite(edge03);
  const Index edge32 = Owner::opposite(edge23);

  owner._edges[std::size_t(edge)].prev = edge12;
  owner._edges[std::size_t(edge)].next = edge10;
  owner._edges[std::size_t(edge10)].prev = edge;
  owner._edges[std::size_t(edge12)].next = edge;

  owner._edges[std::size_t(opposite)].prev = edge30;
  owner._edges[std::size_t(opposite)].next = edge32;
  owner._edges[std::size_t(edge32)].prev = opposite;
  owner._edges[std::size_t(edge30)].next = opposite;

  owner._edges[std::size_t(edge)].vertex = owner.origin(edge10);
  owner._edges[std::size_t(opposite)].vertex = owner.origin(edge32);
  for (const Index half_edge : {edge, opposite}) {
    owner._edges[std::size_t(half_edge)].boundary = false;
    owner._edges[std::size_t(half_edge)].constrained = false;
    owner._edges[std::size_t(half_edge)].delaunay = false;
  }
}

} // namespace tf::topology::cdt
