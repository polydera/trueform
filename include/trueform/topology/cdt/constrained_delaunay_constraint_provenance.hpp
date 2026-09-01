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
#include "../../core/buffer.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt {

template <typename Index, typename Param>
struct constrained_delaunay_constraint_provenance_record {
  Index input_id;
  Param t0;
  Param t1;
};

/// Constraint provenance indexed by undirected edge pair. The input-id lane
/// carries presence and is the only lane initialized for unconstrained edges;
/// parameters become live when provenance is assigned.
template <typename Index, typename Param>
class constrained_delaunay_constraint_provenance {
public:
  using value_type =
      constrained_delaunay_constraint_provenance_record<Index, Param>;

  auto clear() -> void {
    _input_ids.clear();
    _first.clear();
    _second.clear();
  }

  auto prepare(std::size_t dart_count, Index none) -> void {
    const std::size_t pair_count = (dart_count + 1) / 2;
    if (pair_count <= _input_ids.size())
      return;
    const std::size_t old_size = _input_ids.size();
    _input_ids.reallocate(pair_count);
    _first.reallocate(pair_count);
    _second.reallocate(pair_count);
    std::fill(_input_ids.begin() + std::ptrdiff_t(old_size), _input_ids.end(),
              none);
  }

  auto set(Index edge, Index input_id, Param first, Param second) -> void {
    const std::size_t pair = std::size_t(edge) / 2;
    _input_ids[pair] = input_id;
    if ((std::size_t(edge) & 1) == 0) {
      _first[pair] = first;
      _second[pair] = second;
    } else {
      _first[pair] = second;
      _second[pair] = first;
    }
  }

  auto get(Index edge, Index none) const -> value_type {
    const std::size_t pair = std::size_t(edge) / 2;
    if (pair >= _input_ids.size() || _input_ids[pair] == none)
      return {none, Param(0), Param(0)};
    return (std::size_t(edge) & 1) == 0
               ? value_type{_input_ids[pair], _first[pair], _second[pair]}
               : value_type{_input_ids[pair], _second[pair], _first[pair]};
  }

private:
  tf::buffer<Index> _input_ids;
  tf::buffer<Param> _first;
  tf::buffer<Param> _second;
};

} // namespace tf::topology::cdt
