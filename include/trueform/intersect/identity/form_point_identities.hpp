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
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/make_equivalence_class_map.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/topo_type.hpp"
#include "../classify/intersection_payload.hpp"
#include "../records/tagged_intersection.hpp"
#include "./form_edge_classes.hpp"
#include "./identify_vertices.hpp"
#include "./identity_records.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace tf::intersect {

/// Name every record's point canonically and publish the tables that
/// describe those names.
///
/// A point is either an original vertex — any record with a vertex
/// target sits on one, and identified vertices already share a
/// canonical id — or a class on a canonical carrier. The two spaces
/// resolve in one union-find over `[vertex keys | carrier classes]`:
/// carrier classes at parameter 0 or 1 join their endpoint vertex, an
/// edge-edge record joins the classes it created on both carriers, and
/// a vertex lying on an edge joins the class it takes on that carrier.
/// Because every vertex key is scanned before every carrier class, the
/// class ids the collapse mints are already partitioned — kind V first,
/// kind E after — which is the id layout this class publishes.
///
/// `parameters` is the kernels' own table, indexed by a record's id: the
/// point's exact position on each edge it lies on, stated when the
/// incidence was decided. A candidate carries the two tickets that read
/// it — which fraction is this carrier's, and which way round.
template <typename Index, typename Int>
auto form_point_identities(
    tf::buffer<tf::intersect::tagged_intersection<Index>> &records,
    const tf::buffer<Index> &vertex_offsets,
    const tf::buffer<std::array<Index, 2>> &identifications,
    const tf::buffer<tf::exact::edge_fractions<Int, Index>> &parameters,
    identity_scratch<Index, Int> &scratch, point_identities<Index, Int> &out,
    bool with_edge_splits = true) -> void {
  out.clear();
  if (records.size() == 0)
    return;

  const Index no_endpoint = vertex_offsets[vertex_offsets.size() - 1];

  // The payload states every identity the kernels knew: the fact's
  // vertices and edges as flat ids, in carrier-key slot order. Nothing
  // here reads a face.
  auto instance_of = [&](const tf::exact::edge_fractions<Int, Index> &ends,
                         int k) -> edge_instance<Index> {
    const auto from = ends.from[std::size_t(k)];
    const auto to = ends.to[std::size_t(k)];
    const auto ca = canonical_flat_vertex(identifications, from);
    const auto cb = canonical_flat_vertex(identifications, to);
    const auto carrier =
        ca < cb ? home_edge<Index>{ca, cb} : home_edge<Index>{cb, ca};
    // The fraction is stated from the edge's lower flat id; the carrier
    // runs from the lower canonical one.
    return {carrier, from, to, (from < to ? ca : cb) != carrier.u};
  };

  // Pre-duplication a record IS its fact: the machine runs here, once
  // per emission, and the duplicator's fan propagates the names.
  auto &incidences = scratch.incidences;
  auto &candidates = scratch.candidates;
  incidences.clear();
  candidates.clear();
  tf::generic_generate(
      tf::make_sequence_range(Index(records.size())),
      std::tie(incidences, candidates), [&](Index row, auto &bufs) {
        auto &[incidence_buf, candidate_buf] = bufs;
        const auto &rec = records[std::size_t(row)];
        const auto &ends = parameters[std::size_t(rec.id)];
        const bool at_vertex =
            rec.target.label == tf::topo_type::vertex ||
            rec.target_other.label == tf::topo_type::vertex;
        const bool on_edge =
            rec.target.label == tf::topo_type::edge ||
            rec.target_other.label == tf::topo_type::edge;
        if (at_vertex) {
          // A vertex-and-edge fact carries its vertex in slot 1; the
          // vertex is also a split of the edge and enters the carrier's
          // order, and the equivalence pulls the class onto the vertex.
          const auto vid = on_edge ? ends.from[1] : ends.from[0];
          incidence_buf.push_back(
              {canonical_flat_vertex(identifications, vid), row});
          if (on_edge) {
            const auto e = instance_of(ends, 0);
            candidate_buf.push_back({e.carrier, row, std::int8_t(1),
                                     std::int8_t(0),
                                     std::int8_t(e.reversed)});
          }
          return;
        }
        if (!on_edge)
          return;
        // Payload slots are already in carrier-key order, so the slot is
        // the fraction index.
        const auto e0 = instance_of(ends, 0);
        candidate_buf.push_back({e0.carrier, row, std::int8_t(0),
                                 std::int8_t(0), std::int8_t(e0.reversed)});
        if (rec.target.label == tf::topo_type::edge &&
            rec.target_other.label == tf::topo_type::edge) {
          const auto e1 = instance_of(ends, 1);
          candidate_buf.push_back({e1.carrier, row, std::int8_t(1),
                                   std::int8_t(1), std::int8_t(e1.reversed)});
        }
      },
      tf::checked(24000));

  auto &classes = scratch.classes;
  auto &candidate_class = scratch.candidate_class;
  form_edge_classes<Index, Int>(records, parameters, no_endpoint, scratch,
                                classes);
  const auto n_classes = classes.endpoint.size();

  auto &vertex_keys = scratch.vertex_keys;
  vertex_keys.allocate(incidences.size() + n_classes);
  tf::parallel_for_each(
      tf::make_sequence_range(Index(incidences.size())),
      [&](Index i) {
        vertex_keys[std::size_t(i)] = incidences[std::size_t(i)].vertex;
      },
      tf::checked);
  tf::parallel_for_each(
      tf::make_sequence_range(Index(n_classes)),
      [&, base = incidences.size()](Index c) {
        vertex_keys[base + std::size_t(c)] = classes.endpoint[std::size_t(c)];
      },
      tf::checked);
  tbb::parallel_sort(vertex_keys.begin(), vertex_keys.end());
  vertex_keys.erase_till_end(
      std::unique(vertex_keys.begin(), vertex_keys.end()));
  // `no_endpoint` is the flat vertex count, so the non-endpoint filler
  // sorts last and is the only entry that can be out of range.
  if (vertex_keys.size() != 0 &&
      vertex_keys[vertex_keys.size() - 1] == no_endpoint)
    vertex_keys.erase_till_end(vertex_keys.end() - 1);

  const auto n_keys = vertex_keys.size();
  const auto key_base = Index(n_keys);
  const Index domain = Index(n_keys + n_classes);
  auto key_index = [&](Index flat) -> Index {
    return Index(
        std::lower_bound(vertex_keys.begin(), vertex_keys.end(), flat) -
        vertex_keys.begin());
  };

  auto &row_class = scratch.row_class;
  row_class.allocate(records.size());
  tf::parallel_fill(row_class, domain);

  tf::parallel_for_each(
      incidences,
      [&](const vertex_incidence<Index> &e) {
        row_class[std::size_t(e.row)] = key_index(e.vertex);
      },
      tf::checked);
  tf::parallel_for_each(
      tf::make_sequence_range(Index(candidates.size())),
      [&](Index i) {
        const auto &candidate = candidates[std::size_t(i)];
        if (candidate.slot == 0)
          row_class[std::size_t(candidate.row)] =
              key_base + candidate_class[std::size_t(i)];
      },
      tf::checked);

  // A record fills its second slot at most once, so the slot-1
  // candidates ARE the equivalences between the two spaces.
  auto &pairs = scratch.pairs;
  pairs.clear();
  tf::generic_generate(tf::make_sequence_range(Index(candidates.size())), pairs,
                       [&](Index i, auto &buf) {
                         const auto &candidate = candidates[std::size_t(i)];
                         if (candidate.slot == 1)
                           buf.push_back(
                               {row_class[std::size_t(candidate.row)],
                                key_base + candidate_class[std::size_t(i)]});
                       },
                       tf::checked(24000));
  tf::generic_generate(tf::make_sequence_range(Index(n_classes)), pairs,
                       [&](Index c, auto &buf) {
                         const auto endpoint = classes.endpoint[std::size_t(c)];
                         if (endpoint != no_endpoint)
                           buf.push_back({key_base + c, key_index(endpoint)});
                       },
                       tf::checked(24000));
  auto &point_of = scratch.point_of;
  point_of.allocate(std::size_t(domain));
  Index n_points;
  if (pairs.size() == 0) {
    // No cross-space equivalences: every key and class is its own
    // point, already in published order.
    for (Index i = 0; i < domain; ++i)
      point_of[std::size_t(i)] = i;
    n_points = domain;
  } else {
    n_points = tf::make_dense_equivalence_class_map(pairs, point_of);
  }

  Index n_vertex_points = 0;
  for (std::size_t k = 0; k < n_keys; ++k)
    if (point_of[k] >= n_vertex_points)
      n_vertex_points = point_of[k] + 1;

  out.vertex_anchors.allocate(std::size_t(n_vertex_points));
  Index minted = 0;
  for (std::size_t k = 0; k < n_keys; ++k) {
    if (point_of[k] != minted)
      continue;
    const auto flat = vertex_keys[k];
    const auto tag = int(std::upper_bound(vertex_offsets.begin(),
                                          vertex_offsets.end(), flat) -
                         vertex_offsets.begin()) -
                     1;
    out.vertex_anchors[std::size_t(minted++)] = {
        std::int16_t(tag), flat - vertex_offsets[std::size_t(tag)]};
  }

  out.home_edges.allocate(std::size_t(n_points - n_vertex_points));
  out.parameters.allocate(std::size_t(n_points - n_vertex_points));
  minted = 0;
  for (std::size_t g = 0; g + 1 < classes.offsets.size(); ++g)
    for (auto c = classes.offsets[g]; c < classes.offsets[g + 1]; ++c) {
      if (point_of[std::size_t(key_base + c)] != n_vertex_points + minted)
        continue;
      out.home_edges[std::size_t(minted)] = classes.carriers[g];
      out.parameters[std::size_t(minted)] = classes.parameter[std::size_t(c)];
      ++minted;
    }

  // Swap, not move: the buffers' capacities circulate between the
  // published tables and the scratch instead of dying each build.
  // The per-carrier split lists exist for the quantum-merge tier that
  // walks carriers; a consumer that closes coincidences on positions
  // instead never reads them, and asks for them not to be built.
  if (with_edge_splits) {
  std::swap(out.edge_carriers, classes.carriers);
  std::swap(out.edge_splits.offsets_buffer(), classes.offsets);
  out.edge_splits.data_buffer().allocate(n_classes);
  tf::parallel_for_each(
      tf::make_sequence_range(Index(n_classes)),
      [&](Index c) {
        out.edge_splits.data_buffer()[std::size_t(c)] =
            point_of[std::size_t(key_base + c)];
      },
      tf::checked);
  }

  tf::parallel_for_each(
      tf::make_sequence_range(Index(records.size())),
      [&](Index row) {
        const auto combined = row_class[std::size_t(row)];
        if (combined != domain)
          records[std::size_t(row)].id = point_of[std::size_t(combined)];
      },
      tf::checked);
}

} // namespace tf::intersect
