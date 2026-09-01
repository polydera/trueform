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
#include "./append_constrained_delaunay_crossing_point.hpp"
#include "./clear_constrained_delaunay_constraint.hpp"
#include "./compute_constrained_delaunay_crossing_parameters.hpp"
#include "./constrained_delaunay_edge_constraint_provenance.hpp"
#include "./constrained_delaunay_vertex_parameter.hpp"
#include "./find_constrained_delaunay_edge.hpp"
#include "./find_constrained_delaunay_obstruction.hpp"
#include "./find_initial_constrained_delaunay_triangle.hpp"
#include "./insert_constrained_delaunay_vertex.hpp"
#include "./mark_constrained_delaunay_edge.hpp"
#include "./note_constrained_delaunay_constraint_provenance.hpp"
#include "./rebase_constrained_delaunay_parameter.hpp"
#include "./remove_constrained_delaunay_cavity.hpp"
#include "./retriangulate_constrained_delaunay_cavity.hpp"
#include "./reuse_constrained_delaunay_edge.hpp"
#include <cstddef>
#include <initializer_list>

namespace tf::topology::cdt {

template <typename Owner>
auto recover_constrained_delaunay_edge(Owner &owner,
                                       typename Owner::index_type first_vertex,
                                       typename Owner::index_type second_vertex,
                                       bool is_boundary,
                                       typename Owner::index_type input_id,
                                       typename Owner::param_type first_param,
                                       typename Owner::param_type second_param,
                                       int depth, bool retried_from_far_end)
    -> bool {
  using Index = typename Owner::index_type;
  using Param = typename Owner::param_type;
  using ObstructionKind = typename Owner::obstruction_kind;
  if (depth > Owner::max_resolution_depth)
    return false;

  const Index existing =
      find_constrained_delaunay_edge(owner, first_vertex, second_vertex);
  if (existing != Owner::none) {
    if (input_id != Owner::none &&
        owner._edges[std::size_t(existing)].constrained !=
            Owner::unconstrained) {
      const auto prior =
          constrained_delaunay_edge_constraint_provenance(owner, existing);
      if (prior.input_id != Owner::none && prior.input_id != input_id)
        owner._constraint_collisions.push_back(
            {existing, prior.input_id, input_id});
    }
    mark_constrained_delaunay_edge(owner, existing, is_boundary);
    note_constrained_delaunay_constraint_provenance(owner, existing, input_id,
                                                    first_param, second_param);
    return true;
  }

  const auto obstruction = owner._allow_crossing_splits
                               ? find_constrained_delaunay_obstruction(
                                     owner, first_vertex, second_vertex)
                               : typename Owner::obstruction{};
  if (obstruction.kind != ObstructionKind::none) {
    if (obstruction.kind != ObstructionKind::crossing) {
      Index vertex = obstruction.vertex;
      if (vertex == Owner::none) {
        for (const Index candidate :
             {owner.origin(obstruction.edge), owner.target(obstruction.edge)}) {
          if (candidate == first_vertex || candidate == second_vertex)
            continue;
          const Param parameter = constrained_delaunay_vertex_parameter(
              owner, first_vertex, second_vertex, candidate);
          if (parameter > Param(0)) {
            vertex = candidate;
            break;
          }
        }
      }
      if (vertex == Owner::none)
        return false;
      const Param parameter = rebase_constrained_delaunay_parameter<Owner>(
          constrained_delaunay_vertex_parameter(owner, first_vertex,
                                                second_vertex, vertex),
          first_param, second_param);
      if (parameter < Param(0))
        return false;
      if (input_id != Owner::none)
        owner._incremental_crossings.push_back(
            {vertex, input_id, Owner::none, parameter, Param(0)});
      return recover_constrained_delaunay_edge(
                 owner, first_vertex, vertex, is_boundary, input_id,
                 first_param, parameter, depth + 1, false) &&
             recover_constrained_delaunay_edge(owner, vertex, second_vertex,
                                               is_boundary, input_id, parameter,
                                               second_param, depth + 1, false);
    }

    const Index crossed = obstruction.edge;
    const Index crossed_first = owner.origin(crossed);
    const Index crossed_second = owner.target(crossed);
    const bool crossed_is_boundary =
        owner._edges[std::size_t(crossed)].constrained ==
        Owner::boundary_constrained;
    const auto other =
        constrained_delaunay_edge_constraint_provenance(owner, crossed);
    const Index other_id = other.input_id;
    const auto local_parameters =
        compute_constrained_delaunay_crossing_parameters(
            owner, first_vertex, second_vertex, crossed_first, crossed_second);
    if (!local_parameters.valid())
      return false;

    const Param self_parameter = rebase_constrained_delaunay_parameter<Owner>(
        local_parameters.self, first_param, second_param);
    const Param other_parameter = rebase_constrained_delaunay_parameter<Owner>(
        local_parameters.other, other.t0, other.t1);
    const auto created = append_constrained_delaunay_crossing_point(
        owner, first_vertex, second_vertex, local_parameters.self);
    const Index crossing = created.vertex;
    if (crossing != Owner::none)
      insert_constrained_delaunay_vertex(owner, crossing, created.location);

    // THE MEETING IS AN IDENTITY WHETHER OR NOT IT IS A NEW POINT. Žiga:
    // "The resolver says, here is a crossing, but that does not exist."
    // What does not exist is the POINT — its lattice coordinate is already
    // `created.coincident`'s. The crossing does exist: it is where these
    // two constraints meet, and `self_parameter` and `other_parameter`
    // place it exactly on each of them. That is the same truth an ordinary
    // mint carries, whose materialized coordinate is also off both exact
    // lines, by under a unit.
    //
    // So one statement serves both, and the existing vertex is the
    // coordinate this one borrows. It cannot instead be stated as a LANDING
    // per side: a landing asserts the vertex lies ON that constraint, and
    // the coincident vertex fails the on-segment predicate for exactly one
    // of the two, every time it arises. Placing the meeting by parameter is
    // what makes it sayable without lying about either side.
    const Index meeting =
        crossing != Owner::none ? crossing : created.coincident;
    if (meeting != Owner::none && input_id != Owner::none &&
        other_id != Owner::none && input_id != other_id)
      owner._incremental_crossings.push_back(
          {meeting, input_id, other_id, self_parameter, other_parameter});

    if (crossing == Owner::none) {
      const Index coincident = created.coincident;
      if (coincident == Owner::none)
        return false;
      if (coincident == crossed_first || coincident == crossed_second) {
        const Param parameter = rebase_constrained_delaunay_parameter<Owner>(
            constrained_delaunay_vertex_parameter(owner, first_vertex,
                                                  second_vertex, coincident),
            first_param, second_param);
        if (parameter < Param(0))
          return false;
        return recover_constrained_delaunay_edge(
                   owner, first_vertex, coincident, is_boundary, input_id,
                   first_param, parameter, depth + 1, false) &&
               recover_constrained_delaunay_edge(
                   owner, coincident, second_vertex, is_boundary, input_id,
                   parameter, second_param, depth + 1, false);
      }

      const Param parameter = rebase_constrained_delaunay_parameter<Owner>(
          constrained_delaunay_vertex_parameter(owner, crossed_first,
                                                crossed_second, coincident),
          other.t0, other.t1);
      if (parameter < Param(0))
        return false;
      const Index stale =
          find_constrained_delaunay_edge(owner, crossed_first, crossed_second);
      if (stale != Owner::none)
        clear_constrained_delaunay_constraint(owner, stale);
      return recover_constrained_delaunay_edge(
                 owner, crossed_first, coincident, crossed_is_boundary,
                 other_id, other.t0, parameter, depth + 1, false) &&
             recover_constrained_delaunay_edge(
                 owner, coincident, crossed_second, crossed_is_boundary,
                 other_id, parameter, other.t1, depth + 1, false) &&
             recover_constrained_delaunay_edge(
                 owner, first_vertex, second_vertex, is_boundary, input_id,
                 first_param, second_param, depth + 1, false);
    }

    const Index stale =
        find_constrained_delaunay_edge(owner, crossed_first, crossed_second);
    if (stale != Owner::none)
      clear_constrained_delaunay_constraint(owner, stale);
    return recover_constrained_delaunay_edge(
               owner, crossed_first, crossing, crossed_is_boundary, other_id,
               other.t0, other_parameter, depth + 1, false) &&
           recover_constrained_delaunay_edge(
               owner, crossing, crossed_second, crossed_is_boundary, other_id,
               other_parameter, other.t1, depth + 1, false) &&
           recover_constrained_delaunay_edge(
               owner, first_vertex, crossing, is_boundary, input_id,
               first_param, self_parameter, depth + 1, false) &&
           recover_constrained_delaunay_edge(
               owner, crossing, second_vertex, is_boundary, input_id,
               self_parameter, second_param, depth + 1, false);
  }

  Index initial = Owner::none;
  Index clockwise_vertex = Owner::none;
  Index counterclockwise_vertex = Owner::none;
  Index blocking = Owner::none;
  if (!find_initial_constrained_delaunay_triangle(
          owner, first_vertex, second_vertex, initial, clockwise_vertex,
          counterclockwise_vertex, &blocking)) {
    if (owner._allow_crossing_splits && blocking != Owner::none) {
      const Param parameter = rebase_constrained_delaunay_parameter<Owner>(
          constrained_delaunay_vertex_parameter(owner, first_vertex,
                                                second_vertex, blocking),
          first_param, second_param);
      if (parameter >= Param(0)) {
        if (input_id != Owner::none)
          owner._incremental_crossings.push_back(
              {blocking, input_id, Owner::none, parameter, Param(0)});
        return recover_constrained_delaunay_edge(
                   owner, first_vertex, blocking, is_boundary, input_id,
                   first_param, parameter, depth + 1, false) &&
               recover_constrained_delaunay_edge(
                   owner, blocking, second_vertex, is_boundary, input_id,
                   parameter, second_param, depth + 1, false);
      }
    }
    if (owner._allow_crossing_splits && !retried_from_far_end)
      return recover_constrained_delaunay_edge(
          owner, second_vertex, first_vertex, is_boundary, input_id,
          second_param, first_param, depth + 1, true);
    return false;
  }

  owner._vertices_cw.clear();
  owner._vertices_ccw.clear();
  owner._deleted_edges.clear();
  Index passed_through = Owner::none;
  if (!remove_constrained_delaunay_cavity(owner, first_vertex, second_vertex,
                                          clockwise_vertex,
                                          counterclockwise_vertex, initial,
                                          &passed_through)) {
    // A VERTEX ON THE CONSTRAINT'S INTERIOR IS A SPLIT THIS CONSTRAINT DOES
    // NOT YET OWN. The cavity walk found the corridor has no interior at
    // that vertex, so no cavity exists to carve and the recovery cannot
    // finish here — the fan is already part-dismantled, so this build ends.
    // What it must NOT do is end SILENTLY: a refusal that states nothing
    // leaves the next round the identical constraint, and the wave restates
    // it forever. Stating the split is what makes the next build's corridor
    // well formed, and it is the same statement the initial-triangle walk
    // makes when a vertex blocks it.
    if (owner._allow_crossing_splits && passed_through != Owner::none &&
        input_id != Owner::none) {
      const Param parameter = rebase_constrained_delaunay_parameter<Owner>(
          constrained_delaunay_vertex_parameter(owner, first_vertex,
                                                second_vertex, passed_through),
          first_param, second_param);
      if (parameter >= Param(0))
        owner._incremental_crossings.push_back(
            {passed_through, input_id, Owner::none, parameter, Param(0)});
    }
    return false;
  }

  const Index before_first = find_constrained_delaunay_edge(
      owner, first_vertex, owner._vertices_ccw[1]);
  const Index before_second = find_constrained_delaunay_edge(
      owner, second_vertex,
      owner._vertices_cw[owner._vertices_cw.size() - std::size_t(2)]);
  const Index constrained = reuse_constrained_delaunay_edge(
      owner, first_vertex, second_vertex, before_first, before_second,
      owner._deleted_edges.back());
  owner._deleted_edges.pop_back();
  mark_constrained_delaunay_edge(owner, constrained, is_boundary);
  note_constrained_delaunay_constraint_provenance(owner, constrained, input_id,
                                                  first_param, second_param);

  return retriangulate_constrained_delaunay_cavity(owner, owner._vertices_cw,
                                                   true) &&
         retriangulate_constrained_delaunay_cavity(owner, owner._vertices_ccw,
                                                   false);
}

} // namespace tf::topology::cdt
