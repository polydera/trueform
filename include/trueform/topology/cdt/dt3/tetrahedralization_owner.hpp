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
#include "../../../core/blocked_buffer.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/index_map.hpp"
#include "../../../exact/vertex.hpp"
#include "../../tetrahedralization_refusal.hpp"
#include "../../tetrahedralization_stats.hpp"
#include "./tet_cell_key.hpp"
#include "./tet_side.hpp"
#include "./tet_site.hpp"
#include "./tet_site_domain.hpp"
#include "./tet_site_key.hpp"
#include "./tet_state.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>

namespace tf::topology::cdt::dt3 {

/// Flat cell lanes share one index space; facet tickets are 4 * cell + slot.
/// Finite cells have positive exact orientation. Hull cells carry infinity in
/// corner three, close every facet, and admit sites by visibility or their
/// finite mate's empty-sphere predicate on the hull plane. Only empty-sphere
/// ties use symbolic ranks; orientation is never perturbed.
template <typename Index, typename Coord, typename Int>
struct tetrahedralization_owner {
  using index_type = Index;
  using coord_type = Coord;
  using int_type = Int;
  using site_type = tf::exact::vertex<Index, Coord>;

  /// No neighbour, and the non-site corner every hull cell carries. Neither
  /// is ever a site id, a coordinate or a Simulation of Simplicity key.
  static constexpr Index none = Index(-1);
  static constexpr Index infinite = Index(-2);
  /// A facet ticket is `4 * cell + slot`, so the cell space is a quarter of
  /// the index space and a site space is the whole of it.
  static constexpr Index max_tets = std::numeric_limits<Index>::max() / 4;
  static constexpr Index max_sites = std::numeric_limits<Index>::max();
  /// A compaction is due once the dead slots are one part in this many of
  /// the cell space.
  static constexpr std::size_t compaction_dead_divisor = 2;

  tf::buffer<site_type> _sites;
  tf::buffer<tet_site<Index, Coord>> _site_scratch;
  tf::index_map_buffer<Index> _index_map;
  tf::buffer<Index> _order;
  tf::buffer<std::uint64_t> _priorities;
  /// The halved extent of every site, which the scheduling keys are cut on.
  tet_site_domain<Coord> _domain{};
  tf::buffer<std::uint64_t> _keys;
  tf::buffer<tet_site_key<Index>> _key_records;
  tf::buffer<tet_site_key<Index>> _key_scratch;
  tf::buffer<std::size_t> _counts;
  tf::buffer<std::size_t> _block_offsets;

  tf::blocked_buffer<Index, 4> _corners;
  tf::blocked_buffer<Index, 4> _neighbors;
  tf::buffer<tet_state> _states;

  tf::blocked_buffer<Index, 4> _corner_scratch;
  tf::blocked_buffer<Index, 4> _neighbor_scratch;
  tf::buffer<tet_state> _state_scratch;
  tf::buffer<Index> _compaction;
  tf::buffer<tet_cell_key<Index>> _cell_keys;

  tf::buffer<Index> _cavity;
  tf::buffer<Index> _boundary;
  tf::buffer<tet_side<Index>> _sides;

  Index _hint = 0;
  /// The first name above every site's: the counter a caller minting sites
  /// for an append draws from.
  std::size_t _next_name = 0;
  std::size_t _n_dead = 0;
  /// Live finite cells, which a compaction places first. Only a compaction
  /// states it, so it answers for the complex a finished build published.
  std::size_t _n_finite = 0;
  tf::tetrahedralization_refusal _refusal =
      tf::tetrahedralization_refusal::none;
  tf::tetrahedralization_stats _stats;
};

} // namespace tf::topology::cdt::dt3
