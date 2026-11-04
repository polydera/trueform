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
#include "./tet_cavity_links.hpp"
#include "./tet_cell_key.hpp"
#include "./tet_deep_site.hpp"
#include "./tet_site.hpp"
#include "./tet_site_cells.hpp"
#include "./tet_state.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

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
  /// The owner of a facet two faces own: a fold's child, both named in
  /// `_shared`.
  static constexpr Index shared_owner = Index(-3);
  static_assert(infinite >= tet_cavity_links<Index>::lowest_corner,
                "a corner addresses its cavity-link stamp");
  /// A facet ticket is `4 * cell + slot`, so the cell space is a quarter of
  /// the index space and a site space is the whole of it.
  static constexpr Index max_tets = std::numeric_limits<Index>::max() / 4;
  static constexpr Index max_sites = std::numeric_limits<Index>::max();
  /// A compaction is due once the dead slots are one part in this many of
  /// the cell space.
  static constexpr std::size_t compaction_dead_divisor = 2;

  /// Per slot the site's name and its floor on the frame; and per site that
  /// stood below the frame a row by name, ascending, with `_deepest` the
  /// deepest depth a row states: no site has stood below it, and it is zero
  /// while every site has stood on the frame. A withdrawn site's row stays,
  /// named by no slot. The weld, the append and the withdrawal write the
  /// floors, the append states the rows (@ref state_tet_deep_site) and the
  /// clear empties all three; every other reader takes them through their
  /// one reader (@ref tet_site_lanes).
  tf::buffer<site_type> _floors;
  tf::buffer<tet_deep_site<Index>> _deep;
  int _deepest = 0;
  /// A cell holding each site (@ref tet_site_cells), kept only while
  /// `_record_site_cells` is set (@ref record_tet_site_cells); a clear drops
  /// both.
  tet_site_cells<Index> _site_cells;
  bool _record_site_cells = false;
  tf::buffer<tet_site<Index, Coord>> _site_scratch;
  tf::buffer<tet_site<Index, Coord>> _weld_scratch;
  tf::buffer<std::pair<std::uint64_t, Index>> _appended_keys;
  tf::index_map_buffer<Index> _index_map;
  tf::buffer<Index> _order;
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
  tf::buffer<tet_cell_key<Index>> _cell_key_scratch;
  tf::buffer<Index> _key_offsets;
  tf::buffer<Index> _block_counts;

  /// The facets no cavity crosses, stated where the flood reads them: bit
  /// `slot` of a row's byte is set when its facet opposite `slot` is
  /// protected, on both cells of the facet. Rows past the lane's end hold no
  /// bit, and every writer of a row states its byte (@ref
  /// protect_tet_facets). A cavity that reaches both cells of a protected
  /// facet wraps it, and no star keeps the facet: the flood gives that
  /// cavity back and the append withdraws its site, leaving the region to
  /// the operation that supplied it.
  tf::buffer<std::uint8_t> _protected_slots;
  tf::buffer<std::uint8_t> _protected_scratch;
  /// The owner lane beside the bits: per row four faces, the face whose
  /// child the protected facet at that slot is, `none` where unprotected or
  /// owned by no face. Sized with the bits, carried where they are carried.
  tf::buffer<Index> _owners;
  tf::buffer<Index> _owner_scratch;
  /// The facets more than one face owns — a stack: coplanar faces lying on
  /// each other, a fold's two or a surface's sheets — one row per owner,
  /// corner slots ascending, sorted by facet then face
  /// (@ref tet_shared_owners, @ref share_tet_facet).
  tf::buffer<std::pair<std::array<Index, 3>, Index>> _shared;
  /// Protected facets the complex does not hold, corner slots ascending and
  /// sorted, each with its owner: they bar nothing, and they count.
  tf::buffer<std::pair<std::array<Index, 3>, Index>> _protected_unheld;
  /// The protected facets, held or not; zero is the stock complex.
  std::size_t _n_protected = 0;
  /// The slots an append withdrew because their cavities wrapped a
  /// protected facet.
  tf::buffer<Index> _withdrawn;
  /// The protected facets cavities wrapped, corner slots ascending, until
  /// the operation that hands their regions on reads and clears them.
  tf::buffer<std::array<Index, 3>> _wraps;

  /// The edges the cavities of the appends since a recovery operation last
  /// cleared them took, each its corner slots ascending, recorded only
  /// while `_record_killed` is set: the dirty set of a recovery round, read
  /// off the insertions' own product.
  tf::buffer<std::array<Index, 2>> _killed;
  bool _record_killed = false;
  /// The finite rows written since the operation that reads them cleared
  /// them, in writing order, recorded only while `_record_written` is set:
  /// the only rows whose facets can have stopped being locally Delaunay.
  tf::buffer<Index> _written;
  bool _record_written = false;

  tf::buffer<Index> _cavity;
  tf::buffer<Index> _boundary;
  /// The rows a star writes: the cavity's dead rows first, fresh rows after.
  tf::buffer<Index> _rows;
  tet_cavity_links<Index> _links;

  Index _hint = 0;
  /// The first name above every site's: the counter a caller minting sites
  /// for an append draws from.
  std::size_t _next_name = 0;
  std::size_t _n_dead = 0;
  /// Live finite cells, which a compaction places first. Only a compaction
  /// states it, so it answers for the complex a finished build published.
  std::size_t _n_finite = 0;
  /// Whether every cell is Delaunay: so for the complex a build publishes,
  /// not from the first 2-3 flip (@ref flip_tet_facet) on, after which a
  /// walk is bounded and a flood rechecks what its site sees.
  bool _delaunay = true;
  tf::tetrahedralization_refusal _refusal =
      tf::tetrahedralization_refusal::none;
  tf::tetrahedralization_stats _stats;
};

} // namespace tf::topology::cdt::dt3
