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
#include "../core/index_map.hpp"
#include "../core/none.hpp"
#include "../core/points.hpp"
#include "../core/range.hpp"
#include "../exact/int32.hpp"
#include "../exact/resolve_int_type.hpp"
#include "../exact/vertex.hpp"
#include "./cdt/delaunay_execution_policy.hpp"
#include "./cdt/dt3/append_tetrahedralization_sites.hpp"
#include "./cdt/dt3/build_parallel_tetrahedralization.hpp"
#include "./cdt/dt3/build_tetrahedralization.hpp"
#include "./cdt/dt3/check_tetrahedralization_validity.hpp"
#include "./cdt/dt3/clear_tet_claim_workspace.hpp"
#include "./cdt/dt3/clear_tetrahedralization.hpp"
#include "./cdt/dt3/tet_claim_workspace.hpp"
#include "./cdt/dt3/tet_owner_view.hpp"
#include "./cdt/dt3/tetrahedralization_owner.hpp"
#include "./tetrahedralization_refusal.hpp"
#include "./tetrahedralization_stats.hpp"
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup topology
/// @brief Delaunay tetrahedralization of lattice sites.
///
/// Builds finite tetrahedra with positive exact orientation. Coincident sites
/// weld to the lowest canonical name; empty-sphere ties use those names.
/// Cell corners index sites(), and index_map() relates sites to input slots.
/// Cells are published ascending by their sorted corner slots, each stated
/// from its smallest corner with its orientation kept, so the product is the
/// same at any worker count and under either execution policy. Refusal
/// publishes no cells and retains any prepared sites and map.
///
/// @tparam Index Integer type naming sites, cells and facets.
/// @tparam Coord Integer type a site coordinate is stored in.
/// @tparam Int Integer lattice the exact predicates dispatch on; defaults to
///   `Coord` itself via @ref tf::exact::resolve_int_type. A caller carrying a
///   scaled lattice stores it in a wider `Coord`, states the lattice, and
///   states its scale against @ref tf::exact::insphere_scale_bits.
/// @tparam ExecutionPolicy Selects automatic parallel or serial scheduling.
template <typename Index = int, typename Coord = tf::exact::int32,
          typename Int = tf::exact::resolve_int_type<tf::none_t, Coord>,
          typename ExecutionPolicy =
              tf::topology::cdt::parallel_delaunay_execution_policy>
class delaunay_tetrahedralizer {
  static_assert(!std::is_floating_point_v<Coord> &&
                    sizeof(Coord) >= sizeof(Int),
                "a site coordinate is stored on the lattice or wider");

  tf::topology::cdt::dt3::tetrahedralization_owner<Index, Coord, Int> _owner;
  std::conditional_t<ExecutionPolicy::parallel,
                     tf::topology::cdt::dt3::tet_claim_workspace<Index>,
                     tf::none_t>
      _workspace;

  auto published() const {
    return tf::topology::cdt::dt3::tet_owner_view<decltype(_owner)>(_owner);
  }

  template <typename Sites> auto build_sites(const Sites &sites) -> bool {
    if constexpr (ExecutionPolicy::parallel)
      return tf::topology::cdt::dt3::build_parallel_tetrahedralization(
          _owner, sites, _workspace);
    else
      return tf::topology::cdt::dt3::build_tetrahedralization(_owner, sites);
  }

public:
  using index_type = Index;
  using int_type = Int;
  using coord_type = Coord;
  using site_type = tf::exact::vertex<Index, Coord>;

  /// No neighbour, which a hull facet of a published cell reads as.
  static constexpr Index k_none =
      tf::topology::cdt::dt3::tetrahedralization_owner<Index, Coord, Int>::none;
  /// The non-site corner every hull cell carries.
  static constexpr Index k_infinite =
      tf::topology::cdt::dt3::tetrahedralization_owner<Index, Coord,
                                                       Int>::infinite;

  /// Clear all output and build state while retaining buffer capacity.
  auto clear() -> void {
    tf::topology::cdt::dt3::clear_tetrahedralization(_owner);
    if constexpr (ExecutionPolicy::parallel)
      tf::topology::cdt::dt3::clear_tet_claim_workspace(_workspace);
  }

  /// Build from lattice points, each site named by its input slot.
  /// Returns false and states @ref refusal() when the welded sites span less
  /// than three dimensions, or input slots or cells outgrow `Index`.
  template <typename PointsPolicy>
  auto build(const tf::points<PointsPolicy> &points) -> bool {
    return build_sites(points);
  }

  /// Build from sites that already carry their canonical names, which the
  /// weld keeps the lowest of and the symbolic tie-break ranks by.
  template <typename Iterator>
  auto build(const tf::range<Iterator, tf::dynamic_size> &sites) -> bool {
    static_assert(
        std::is_same_v<std::decay_t<decltype(*std::declval<Iterator>())>,
                       site_type>,
        "prepared sites must use the triangulator's site type");
    return build_sites(sites);
  }

  /// Append distinct prepared sites with fresh names after a successful build.
  /// Site slots stay fixed; input-map slots append in input order. Position
  /// and name admission belong to the supplying recovery operation. A site
  /// whose cavity would wrap a facet a constraint operation protects is
  /// withdrawn: its input slot maps to @ref k_none and the slots after it in
  /// the call close up.
  template <typename Sites> auto append_sites(const Sites &sites) -> bool {
    return tf::topology::cdt::dt3::append_tetrahedralization_sites<
        ExecutionPolicy::parallel>(_owner, sites, _workspace);
  }

  /// Append as above, serially in the canonical order, withdrawing a site
  /// whenever `keep(owner, i)`, read just before site `i` of `sites` would be
  /// inserted, answers false: the recovery operation's own lazy test.
  template <typename Sites, typename Keep>
  auto append_sites(const Sites &sites, const Keep &keep) -> bool {
    return tf::topology::cdt::dt3::append_tetrahedralization_sites<
        ExecutionPolicy::parallel>(_owner, sites, _workspace, keep);
  }

  /// The owner itself, for the flips a recovery makes on its lanes.
  auto owner() -> tf::topology::cdt::dt3::tetrahedralization_owner<Index, Coord,
                                                                 Int> & {
    return _owner;
  }
  auto owner() const
      -> const tf::topology::cdt::dt3::tetrahedralization_owner<Index, Coord,
                                                                 Int> & {
    return _owner;
  }

  /// Whether appends record the edges their cavities take — the dirty set
  /// a recovery operation reads — and the edges recorded since the last
  /// clear, each its corner slots ascending.
  auto record_killed_edges(bool on) -> void {
    _owner._record_killed = on;
    _owner._killed.clear();
  }
  auto killed_edges() const -> const tf::buffer<std::array<Index, 2>> & {
    return _owner._killed;
  }
  auto clear_killed_edges() -> void { _owner._killed.clear(); }

  /// Why the last build or append published no cells.
  auto refusal() const -> tf::tetrahedralization_refusal {
    return _owner._refusal;
  }

  /// Work since the last build, including admitted appends.
  auto stats() const -> const tf::tetrahedralization_stats & {
    return _owner._stats;
  }

  /// The finite cells, four site slots each.
  auto n_tets() const -> std::size_t { return published().n_tets(); }

  auto tets() const { return published().tets(); }

  /// Per finite cell, the cell across the facet opposite each corner, or
  /// @ref k_none where that facet is on the convex hull.
  auto neighbors() const { return published().neighbors(); }

  /// The welded build sites followed by admitted appended sites.
  auto n_sites() const -> std::size_t { return published().n_sites(); }

  auto sites() const { return published().sites(); }

  /// The first name above every site's, from which sites minted for an
  /// append take fresh names.
  auto next_name() const -> std::size_t { return published().next_name(); }

  /// `f()[input]` is the site an input slot welded onto, or @ref k_none for
  /// a withdrawn append, `kept_ids()[site]` the input slot retained for that
  /// site's canonical name.
  auto index_map() const -> const tf::index_map_buffer<Index> & {
    return _owner._index_map;
  }

  /// Whether the published complex satisfies every structural law of a
  /// tetrahedralization. A refused build has none to satisfy.
  auto is_valid() const -> bool {
    return _owner._refusal == tf::tetrahedralization_refusal::none &&
           tf::topology::cdt::dt3::check_tetrahedralization_validity(_owner);
  }
};

} // namespace tf
