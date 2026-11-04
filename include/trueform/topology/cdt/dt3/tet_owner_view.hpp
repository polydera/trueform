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
#include "../../../core/range.hpp"
#include "../../../core/tetras.hpp"
#include "../../../core/views/blocked_range.hpp"
#include "../../../core/views/mapped_range.hpp"
#include "../../../core/views/take.hpp"
#include "./tet_site_position.hpp"
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// The published complex of an owner at one moment, a value of ranges over
/// its lanes: its finite cells, their links with the hull read as `k_none`,
/// its sites and its input map. An append or a compaction may move every
/// lane it reads, so a snapshot is taken where it is read and never held
/// across either; re-take it after every mutation, as a binding re-takes a
/// view after growth. Take it after a canonical compaction.
template <typename Owner> class tet_owner_view {
public:
  using index_type = typename Owner::index_type;
  using int_type = typename Owner::int_type;
  using coord_type = typename Owner::coord_type;
  using site_type = typename Owner::site_type;
  static constexpr index_type k_none = Owner::none;

private:
  static auto sites_of(const Owner &owner) {
    return tet_shallow_sites_of(owner);
  }
  static auto tets_of(const Owner &owner) {
    return tf::make_tetras(
        tf::take(owner._corners.data_buffer(), owner._n_finite * 4));
  }
  static auto neighbors_of(const Owner &owner) {
    const index_type finite = index_type(owner._n_finite);
    index_type boundary = k_none;
    return tf::make_blocked_range<4>(tf::make_mapped_range(
        tf::take(owner._neighbors.data_buffer(), owner._n_finite * 4),
        [finite, boundary](index_type cell) {
          return cell < finite ? cell : boundary;
        }));
  }
  static auto index_map_of(const Owner &owner) {
    return tf::make_index_map(tf::make_range(owner._index_map.f()),
                              tf::make_range(owner._index_map.kept_ids()));
  }

  decltype(sites_of(std::declval<const Owner &>())) _sites;
  decltype(tets_of(std::declval<const Owner &>())) _tets;
  decltype(neighbors_of(std::declval<const Owner &>())) _neighbors;
  decltype(index_map_of(std::declval<const Owner &>())) _index_map;
  std::size_t _n_tets;
  std::size_t _next_name;

public:
  explicit tet_owner_view(const Owner &owner)
      : _sites(sites_of(owner)), _tets(tets_of(owner)),
        _neighbors(neighbors_of(owner)), _index_map(index_map_of(owner)),
        _n_tets(owner._n_finite), _next_name(owner._next_name) {}

  auto sites() const { return _sites; }
  auto n_sites() const -> std::size_t { return _sites.size(); }
  auto next_name() const -> std::size_t { return _next_name; }
  auto index_map() const -> const decltype(_index_map) & { return _index_map; }
  auto n_tets() const -> std::size_t { return _n_tets; }
  auto tets() const { return _tets; }
  auto neighbors() const { return _neighbors; }
};

} // namespace tf::topology::cdt::dt3
