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
#include "../../../core/index_map.hpp"
#include "../../../core/none.hpp"
#include "../../../core/range.hpp"
#include "../../../core/tetras.hpp"
#include "../../../core/views/blocked_range.hpp"
#include "../../../core/views/mapped_range.hpp"
#include "./append_tetrahedralization_sites.hpp"
#include "./tet_facet_is_protected.hpp"
#include "./tet_site_position.hpp"
#include "./tet_split_route.hpp"
#include "./tet_state.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// An owner as a recovery operation reads and grows it, for the length of
/// that operation. Its cells are every row of the owner's lanes, dead rows
/// and hull cells among them, a link to a hull cell reading `k_none`, so it
/// reads the complex between compactions; `tet_finite_cell` says which rows
/// a reader may enter. Every read takes a fresh range over the lanes, so
/// nothing read through it outlives the append that follows.
/// It is a parameter bundle scoped to one operation, never stored beyond
/// it, so the views-are-values law holds by its lifetime.
template <bool Parallel, typename Owner, typename Work>
class tet_owner_complex {
  Owner &_owner;
  Work &_work;

public:
  using index_type = typename Owner::index_type;
  using int_type = typename Owner::int_type;
  using coord_type = typename Owner::coord_type;
  using site_type = typename Owner::site_type;
  static constexpr index_type k_none = Owner::none;

  tet_owner_complex(Owner &owner, Work &work) : _owner(owner), _work(work) {}

  auto n_sites() const -> std::size_t { return tet_site_count(_owner); }
  auto next_name() const -> std::size_t { return _owner._next_name; }
  auto index_map() const -> const tf::index_map_buffer<index_type> & {
    return _owner._index_map;
  }
  auto n_tets() const -> std::size_t {
    return std::size_t(_owner._corners.size());
  }
  auto tets() const { return tf::make_tetras(_owner._corners.data_buffer()); }
  auto neighbors() const {
    const auto corners = tf::make_blocked_range<4>(
        tf::make_range(_owner._corners.data_buffer()));
    index_type boundary = k_none;
    return tf::make_blocked_range<4>(tf::make_mapped_range(
        tf::make_range(_owner._neighbors.data_buffer()),
        [corners, boundary](index_type cell) {
          return corners[std::size_t(cell)][3] == Owner::infinite ? boundary
                                                                  : cell;
        }));
  }
  auto finite(index_type cell) const -> bool {
    return cell >= 0 &&
           std::size_t(cell) < std::size_t(_owner._corners.size()) &&
           _owner._states[std::size_t(cell)] == tet_state::live;
  }
  /// A cell holding `site`, or `k_none` before any did.
  auto site_cell(index_type site) const -> index_type {
    return std::size_t(site) < _owner._site_cells.size()
               ? _owner._site_cells.load(std::size_t(site))
               : k_none;
  }
  auto entry() const -> index_type {
    const auto hint = _owner._hint;
    return _owner._corners[std::size_t(hint)][3] == Owner::infinite
               ? _owner._neighbors[std::size_t(hint)][3]
               : hint;
  }
  template <typename Sites> auto append_sites(const Sites &sites) -> bool {
    return append_tetrahedralization_sites<Parallel>(_owner, sites, _work);
  }
  /// Append serially in the canonical order, withdrawing a site `keep`
  /// declines just before its insertion.
  template <typename Sites, typename Keep>
  auto append_sites(const Sites &sites, const Keep &keep) -> bool {
    return append_tetrahedralization_sites<Parallel>(_owner, sites, _work,
                                                     keep);
  }
  /// Append serially, every site placed by the split of the simplex it lies
  /// in (@ref tet_split_route).
  template <typename Sites>
  auto append_sites(const Sites &sites, tet_split_route_t route) -> bool {
    return append_tetrahedralization_sites<Parallel>(_owner, sites, _work,
                                                     tf::none, route);
  }
  /// The owner itself, for the flips a recovery makes on its lanes.
  auto owner() -> Owner & { return _owner; }
  auto owner() const -> const Owner & { return _owner; }
  /// Whether the facet of `cell` opposite `slot` is protected.
  auto facet_protected(index_type cell, std::size_t slot) const -> bool {
    return tet_facet_is_protected(_owner, cell, slot);
  }
  /// Whether appends record the edges their cavities take, and the edges
  /// recorded since the last clear, each its corner slots ascending.
  auto record_killed_edges(bool on) -> void {
    _owner._record_killed = on;
    _owner._killed.clear();
  }
  auto killed_edges() const -> const tf::buffer<std::array<index_type, 2>> & {
    return _owner._killed;
  }
  auto clear_killed_edges() -> void { _owner._killed.clear(); }
};

} // namespace tf::topology::cdt::dt3
