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
#include "../core/blocked_buffer.hpp"
#include "../core/buffer.hpp"
#include "../core/edges.hpp"
#include "../core/faces.hpp"
#include "../core/index_map.hpp"
#include "../core/none.hpp"
#include "../core/points.hpp"
#include "../core/range.hpp"
#include "../core/views/constant.hpp"
#include "../core/views/mapped_range.hpp"
#include "../exact/resolve_int_type.hpp"
#include "./cdt/abandon_constrained_delaunay.hpp"
#include "./cdt/build_constrained_delaunay_from_prepared_constraints.hpp"
#include "./cdt/build_constrained_delaunay_triangulation.hpp"
#include "./cdt/clear_constrained_delaunay.hpp"
#include "./cdt/constrained_delaunay_face_constrained.hpp"
#include "./cdt/constrained_delaunay_face_constraint_owners.hpp"
#include "./cdt/constrained_delaunay_face_constraints.hpp"
#include "./cdt/constrained_delaunay_owner.hpp"
#include "./cdt/delaunay_execution_policy.hpp"
#include "./cdt/for_each_constrained_delaunay_crossing.hpp"
#include "./cdt/for_each_constrained_delaunay_face.hpp"
#include "./cdt/materialize_constrained_delaunay_regions.hpp"
#include "./cdt/promote_constrained_delaunay_boundaries.hpp"
#include "./cdt_region_mode.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf {

/// @ingroup topology
/// @brief Constrained Delaunay triangulation of 2D point sets.
///
/// Builds an exact-predicate Delaunay triangulation of the cleaned input
/// points, recovers the constraints, and labels the resulting planar
/// regions. The shared core is an exact Guibas-Stolfi divide-and-conquer build
/// over physically Morton-ordered sites with alternating-axis Dwyer splits.
/// Constraint recovery then edits the retained origin rings and restores the
/// constrained Delaunay property with Lawson flips.
///
/// Two constraint modes: `split_constraints = true` resolves crossings
/// against the live triangulation as they are met, so recovery may
/// introduce output points no input carried; `split_constraints = false`
/// preserves constraints verbatim and fails the build if they cross. Repeated
/// boundary marking of one edge TOGGLES its region-separation parity (a
/// zero-width slit crosses the boundary twice, i.e. not at all).
///
/// Input points are converted to `Int`, welded exactly, and exposed
/// through `points()`.
///
/// @tparam Index The index type.
/// @tparam Coord The input coordinate type. Also the type returned by
///   `converted_points()` after deconversion. May be either an integral or a
///   floating-point type.
/// @tparam Int The integer type for exact predicates; defaults per Coord
///   via @ref tf::exact::resolve_int_type (int32 for float, int64 for
///   double, identity for integral Coord).
/// @tparam ExecutionPolicy Serial or separated-domain parallel scheduling for
///   the shared initial Delaunay operation.
template <typename Index, typename Coord,
          typename Int = tf::exact::resolve_int_type<tf::none_t, Coord>,
          typename ExecutionPolicy =
              tf::topology::cdt::serial_delaunay_execution_policy>
class constrained_delaunay_triangulator {
  using owner_type =
      tf::topology::cdt::constrained_delaunay_owner<Index, Coord, Int,
                                                    ExecutionPolicy>;

  auto owner() -> owner_type & { return _owner; }
  auto owner() const -> const owner_type & { return _owner; }

  owner_type _owner;

public:
  /// The crossing parameter is the geometry here, so it is carried at the
  /// exact substrate's own width rather than squeezed into Index. At 30
  /// bits a crossing near the end of a long constraint quantises to a
  /// displacement of 1/2^30 of its length — millions of lattice units at
  /// pipeline scale. The width is chosen so that a parameter, and the
  /// products the blend and the rebase form from it, stay inside T1/T2.
  using param_t = typename owner_type::param_type;
  static constexpr int k_crossing_param_bits = owner_type::crossing_param_bits;
  static constexpr int k_max_resolution_depth =
      owner_type::max_resolution_depth;

  /// A crossing the incremental resolution created: the point it inserted
  /// and, per crossed constraint, its input id and the parameter of the
  /// crossing along that constraint's whole physical edge.
  using constraint_owner_t = typename owner_type::constraint_provenance;
  using incremental_crossing = typename owner_type::incremental_crossing;
  using constraint_collision = typename owner_type::constraint_collision;

  static constexpr Index k_none = owner_type::none;

  // Constraint state of a half-edge.
  //   k_unconstrained        — flippable, ignored by region labelling.
  //   k_constrained          — preserved by Delaunay restoration; does
  //                            NOT separate regions.
  //   k_boundary_constrained — preserved AND separates regions.
  static constexpr std::uint8_t k_unconstrained = owner_type::unconstrained;
  static constexpr std::uint8_t k_constrained = owner_type::constrained;
  static constexpr std::uint8_t k_boundary_constrained =
      owner_type::boundary_constrained;

  using half_edge = typename owner_type::half_edge;
  using flip_check = typename owner_type::flip_check;

public:
  auto clear() -> void { topology::cdt::clear_constrained_delaunay(owner()); }

  /// @brief Exact-lattice output points indexed by `make_faces()`.
  auto points() const { return tf::make_points(owner()._points); }

  /// Output points deconverted from the exact lattice back to `Coord`.
  auto converted_points() const {
    return tf::make_points(
        tf::make_mapped_range(owner()._points, [this](const auto &p) {
          return owner()._converter.deconvert(p);
        }));
  }

  auto n_triangles() const -> Index {
    return static_cast<Index>(owner()._faces.size());
  }

  /// @brief Visit faces with full adjacency straight off the DCEL:
  /// f(i, v0, v1, v2, label, nbrs, cons): nbrs[k] is the face index
  /// across edge (vk, vk+1) or `k_none`, cons[k] whether that edge is
  /// constrained. Face ids and order match make_faces/region_labels.
  /// Valid after `build_regions()` or a full `build()`.
  template <typename F> auto for_each_face_adjacency(F &&f) const -> void {
    topology::cdt::for_each_constrained_delaunay_face(owner(),
                                                      static_cast<F &&>(f));
  }

  /// @brief The input constraint each edge of face `triangle` belongs to —
  /// edge k spans corners k and k + 1 of `make_faces()[triangle]` — or
  /// `k_none` where the edge is not a constraint.
  ///
  /// This is the identity a crossing report is stated against, and pieces of
  /// a resolved constraint inherit it. Splitting builds and topology phases
  /// prepared for `cdt_region_mode::components` record this carrier; a plain
  /// no-split nesting phase reports `k_none` throughout. Valid after
  /// `build_regions()` or a full `build()`.
  auto face_constraints(Index triangle) const -> std::array<Index, 3> {
    return topology::cdt::constrained_delaunay_face_constraints(owner(),
                                                                triangle);
  }

  /// @brief The full owner record for each edge of face `triangle`.
  ///
  /// Each record carries the input constraint ID and the exact span covered by
  /// this edge, oriented from corner k to corner k + 1. `input_id == k_none`
  /// where the edge is not a constraint. Valid after `build_regions()` or a
  /// full `build()`.
  auto face_constraint_owners(Index triangle) const
      -> std::array<constraint_owner_t, 3> {
    return topology::cdt::constrained_delaunay_face_constraint_owners(owner(),
                                                                      triangle);
  }

  /// Retain constraint ownership on every build mode, including the plain
  /// no-split nesting path. The setting persists across `clear()` and builds.
  auto always_track_constraint_owners() -> void {
    owner()._always_track_constraint_provenance = true;
  }

  /// @brief Whether each edge of face `triangle` is constrained.
  /// Edge k spans corners k and k + 1 of `make_faces()[triangle]`.
  /// Valid after `build_regions()` or a full `build()`.
  auto face_constrained(Index triangle) const -> std::array<bool, 3> {
    return topology::cdt::constrained_delaunay_face_constrained(owner(),
                                                                triangle);
  }

  /// @brief Build and return triangles of the constrained triangulation.
  /// The triangles, in the order `region_labels()` labels them.
  ///
  /// Returns the vertex triples materialized by the triangulation phase.
  auto make_faces() const -> tf::blocked_buffer<Index, 3> {
    return owner()._faces;
  }

  /// Read-only semantic face view.
  auto faces() const { return tf::make_faces(owner()._faces); }

  /// Read-only owning face carrier.
  auto faces_buffer() const -> const tf::blocked_buffer<Index, 3> & {
    return owner()._faces;
  }

  /// @brief Index map between original input point space and the
  /// cleaned output point space.
  ///
  /// `f()[input_id]` is the output index of input point `input_id`.
  /// `kept_ids()[output_id]` is the original input ID that survived the
  /// weld for output slot `output_id`, or the sentinel `f().size()` for a
  /// vertex recovery created. A refused constraint-recovery build retains
  /// this prepared subproduct only when `has_prepared_input_index_map()` is
  /// true.
  auto index_map() const -> const tf::index_map_buffer<Index> & {
    return owner()._index_map;
  }

  /// Whether exact input preparation completed and published `index_map()`.
  /// Constraint recovery may subsequently refuse and abandon topology without
  /// invalidating this input-identity product.
  auto has_prepared_input_index_map() const -> bool {
    return owner()._has_prepared_input_index_map;
  }

  /// @brief Non-const accessor to allow moving the index map out.
  auto index_map() -> tf::index_map_buffer<Index> & {
    return owner()._index_map;
  }

  /// @brief Crossings recovery had to create, as {point, constraint_a,
  ///        constraint_b, t_a, t_b} where each t is a dyadic parameter on
  ///        2^`crossing_param_bits()` along that constraint. The parameter,
  ///        not the coordinate, is the transportable fact: a carrier holding
  ///        the same edge places the split from it exactly.
  auto parameterized_crossings() const
      -> const tf::buffer<incremental_crossing> & {
    return owner()._incremental_crossings;
  }

  static constexpr int crossing_param_bits() { return k_crossing_param_bits; }

  /// @brief Recoveries that found their edge already claimed by a DIFFERENT
  ///        input constraint, as {edge, prior_input, input}: the two inputs
  ///        cover one span. A crossing at a point resolves through
  ///        `parameterized_crossings()`; these are coincidences, and the
  ///        per-edge provenance keeps only the later input.
  auto constraint_collisions() const
      -> const tf::buffer<constraint_collision> & {
    return owner()._constraint_collisions;
  }

  /// @brief Region labels of the walls — the boundary constraints — indexed
  /// by triangle (matches the triangle order produced by `make_faces()`).
  ///
  /// The build's `tf::cdt_region_mode` says which fact these are.
  /// `nesting` propagates a parity: crossing a wall toggles the label, a
  /// plain constraint passes it through, so an odd label is one an odd
  /// number of walls enclose. `components` propagates an id instead: the
  /// walk stops at a wall, label 0 is the hull exterior (which may own no
  /// triangle) and every other component is entered at its lowest triangle,
  /// so the numbering depends on the triangulation alone.
  ///
  /// The ids are what parity cannot express: a loop inside a loop carries
  /// the exterior's parity and never its id. Valid after `build_regions()` or
  /// a full `build()`.
  auto region_labels() const { return tf::make_range(owner()._region_labels); }

  /// The fact currently carried by `region_labels()`.
  auto region_mode() const -> cdt_region_mode { return owner()._region_mode; }

  /// Split-constraints builds only: invoke `f(point, parents)` for
  /// every point lying strictly inside a constraint —
  /// `point` addresses points(), `parents` is the range of input
  /// constraints the point is interior to (one entry = a T-junction at
  /// an existing input vertex; duplicated constraints all appear).
  template <typename F> auto for_each_constraint_crossing(F &&f) const -> void {
    topology::cdt::for_each_constrained_delaunay_crossing(owner(),
                                                          static_cast<F &&>(f));
  }

  /// @brief Build the triangulation and recover the supplied constraints,
  /// without constructing face adjacency or region labels.
  ///
  /// Intersecting constraints are split during constraint recovery. The
  /// resulting output indices refer to `points()`, not necessarily the
  /// original input point IDs.
  ///
  /// `is_boundary` is a per-input-edge mask. An entry of `true` marks the
  /// constraint as a region wall. `false` marks it as
  /// preserved-but-not-boundary. The distinction is retained for a later
  /// `build_regions()` call.
  /// Pass `tf::make_constant_range(true, edges.size())` for the
  /// all-boundary case. `mode` selects the provenance needed by that later
  /// phase without constructing it: components retains input-constraint IDs
  /// even in a no-split build; nesting retains them only when splitting needs
  /// crossing provenance.
  template <typename PointsPolicy, typename EdgesPolicy, typename Iterator,
            std::size_t N>
  auto build_triangulation(const tf::points<PointsPolicy> &pts,
                           const tf::edges<EdgesPolicy> &edges,
                           const tf::range<Iterator, N> &is_boundary,
                           bool split_constraints = true,
                           cdt_region_mode mode = cdt_region_mode::nesting)
      -> bool {
    return topology::cdt::build_constrained_delaunay_triangulation(
        owner(), pts, edges, is_boundary, split_constraints, mode);
  }

  template <typename PointsPolicy, typename EdgesPolicy>
  auto build_triangulation(const tf::points<PointsPolicy> &pts,
                           const tf::edges<EdgesPolicy> &edges,
                           bool split_constraints = true,
                           cdt_region_mode mode = cdt_region_mode::nesting)
      -> bool {
    return build_triangulation(pts, edges,
                               tf::make_constant_range(true, edges.size()),
                               split_constraints, mode);
  }

  /// Build face adjacency and one region-label product for the retained
  /// triangulation. This can be called again with a different mode without
  /// rebuilding or recovering the topology. Relabelling does not retroactively
  /// add constraint provenance omitted by the topology phase.
  auto build_regions(cdt_region_mode mode = cdt_region_mode::nesting) -> void {
    topology::cdt::materialize_constrained_delaunay_regions(owner(), mode);
  }

  /// Promote retained constraints selected by original input-owner id to
  /// region boundaries. Valid after `build_triangulation()` and before
  /// `build_regions()`; false entries retain generic CDT parity. A no-split
  /// nesting build must call `always_track_constraint_owners()` before
  /// triangulation so original input ownership is retained.
  template <typename Promotions>
  auto promote_constraint_boundaries(const Promotions &promotions) -> void {
    topology::cdt::promote_constrained_delaunay_boundaries(owner(),
                                                           promotions);
  }

  /// @brief Build the constrained triangulation and its selected region
  /// labels.
  template <typename PointsPolicy, typename EdgesPolicy, typename Iterator,
            std::size_t N>
  auto build(const tf::points<PointsPolicy> &pts,
             const tf::edges<EdgesPolicy> &edges,
             const tf::range<Iterator, N> &is_boundary,
             bool split_constraints = true,
             cdt_region_mode mode = cdt_region_mode::nesting) -> bool {
    if (!build_triangulation(pts, edges, is_boundary, split_constraints, mode))
      return false;
    build_regions(mode);
    return true;
  }

  /// Discard retained topology and face/region products after a refused
  /// recovery.
  auto abandon() -> void {
    topology::cdt::abandon_constrained_delaunay(owner());
  }

  /// Triangulate the converted points and remapped constraints currently held
  /// by the input-preparation state. A constraint that cannot be recovered
  /// leaves the triangulation half-dismantled, so a refused build reports no
  /// topology or derived products.
  auto build_triangulation_from_constraints() -> bool {
    return topology::cdt::build_constrained_delaunay_from_prepared_constraints(
        owner());
  }

  /// Triangulate retained preparation and materialize the selected regions.
  auto build_from_constraints(cdt_region_mode mode = cdt_region_mode::nesting)
      -> bool {
    if (!build_triangulation_from_constraints())
      return false;
    build_regions(mode);
    return true;
  }

  template <typename PointsPolicy, typename EdgesPolicy>
  auto build(const tf::points<PointsPolicy> &pts,
             const tf::edges<EdgesPolicy> &edges, bool split_constraints = true,
             cdt_region_mode mode = cdt_region_mode::nesting) -> bool {
    return build(pts, edges, tf::make_constant_range(true, edges.size()),
                 split_constraints, mode);
  }
};

} // namespace tf
