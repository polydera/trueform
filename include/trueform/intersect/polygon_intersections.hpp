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
#include "../core/algorithm/block_reduce.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/local_value.hpp"
#include "../core/none.hpp"
#include "../core/polygons.hpp"
#include "../core/range.hpp"
#include "../core/views/offset_block_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "../exact/edge_parameter.hpp"
#include "../exact/resolve_int_type.hpp"
#include "../spatial/policy/tree.hpp"
#include "../topology/policy/face_membership.hpp"
#include "../topology/policy/manifold_edge_link.hpp"
#include "./classify/intersection_payload.hpp"
#include "./face_pairs/face_pair_kernels.hpp"
#include "./face_pairs/face_pair_search.hpp"
#include "./face_pairs/face_self_kernels.hpp"
#include "./identity/form_point_identities.hpp"
#include "./identity/identify_vertices.hpp"
#include "./identity/identity_records.hpp"
#include "./identity/resolve_delivered_vertices.hpp"
#include "./intersect_config.hpp"
#include "./records/coplanar_pair_flags.hpp"
#include "./records/dedup_generator_records.hpp"
#include "./records/dedup_point_deliveries.hpp"
#include "./records/duplicate_tagged_intersection.hpp"
#include "./records/group_intersection_carriers.hpp"
#include "./records/make_pair_group_gate.hpp"
#include "./records/pair_group_gate.hpp"
#include "./records/point_delivery.hpp"
#include "./records/tagged_intersection.hpp"
#include "tbb/task_group.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf {

/// Exact intersection identity of a set of polygon meshes.
///
/// A record's `id` is a canonical point name rather than a slot in a
/// coordinate table, and the tables that define those names are the
/// surface. This class computes no coordinates and keeps none: a point
/// is an original vertex, or an exact parameter class on an original
/// edge, and both are described by generators. Snapping belongs to
/// whoever materializes positions.
///
/// Identity is settled in two tiers. Coincident originals collapse
/// first, so a physical edge has one canonical name however many vertex
/// ids spell it; carriers are then keyed on canonical endpoints, which
/// is what makes duplicated vertices, shared edges across forms, and
/// reversed edge instances one carrier instead of several.
///
/// The classifiers state a point's parameter on the edges it lies on
/// where they decide the incidence, once per fact and before the
/// duplicator fans it, so identity formation reads positions off
/// records instead of reconstructing them from generators. Nothing here
/// builds a coordinate: where an original vertex stands is
/// @ref tf::exact::input_lattice's fact, stated once above this class and
/// handed to every build, and the predicates below are exact.
template <typename Index, typename RealType,
          typename Int = tf::exact::resolve_int_type<tf::none_t, RealType>>
class polygon_intersections {
  using intersection_t = tf::intersect::tagged_intersection<Index>;
  using parameters_t = tf::exact::edge_fractions<Int, Index>;
  using workspace_t =
      tf::intersect::face_pair_workspace<Index, Int, parameters_t>;
  using gate_t = tf::intersect::pair_group_gate<Index>;
  /// One scratch serves both currencies' walks: they share the feature
  /// expansion, and a self record's delivery walk runs the duplication.
  using fan_scratch_t =
      tf::intersect::intersection_fan_scratch<Index, typename gate_t::side>;

public:
  /// The per-carrier split lists serve the quantum-merge tier that
  /// walks carriers. A consumer that closes coincidences on positions
  /// never reads them and says so here, before the build.
  auto with_edge_splits(bool value) -> void { _with_edge_splits = value; }

  /// Self-only build of one form (the `within` bit is implied).
  template <typename Policy, typename Lattice>
  auto build(const tf::polygons<Policy> &form, const Lattice &lattice,
             tf::intersect_config config = {}) -> void {
    assert_form_policies<Policy>();
    clear();
    take_vertex_offsets(lattice);

    const bool primitives = config.mode & tf::intersect_mode::primitives;

    tf::local_value<workspace_t> ws;
    if (primitives)
      run_self_primitives(form, 0, lattice, ws);
    else
      run_self_sos(form, 0, lattice, ws);

    auto apply_to_form = [&form](int, auto &&f) { f(form); };
    finalize_identity(ws, apply_to_form, Index(1), true, !primitives);
  }

  /// Two forms of possibly different policies. Cross records always;
  /// per-form self records when the mode carries `self_intersections`.
  template <typename Policy0, typename Policy1, typename Lattice>
  auto build(const tf::polygons<Policy0> &form0,
             const tf::polygons<Policy1> &form1, const Lattice &lattice,
             tf::intersect_config config = {}) -> void {
    assert_form_policies<Policy0>();
    assert_form_policies<Policy1>();
    clear();
    take_vertex_offsets(lattice);

    const bool primitives = config.mode & tf::intersect_mode::primitives;
    const bool with_self = config.mode & tf::intersect_mode::self_intersections;

    tf::local_value<workspace_t> ws;
    if (primitives) {
      run_primitives_pair(form0, form1, 0, 1, lattice, ws);
      if (with_self) {
        run_self_primitives(form0, 0, lattice, ws);
        run_self_primitives(form1, 1, lattice, ws);
      }
    } else {
      run_sos_pair(form0, form1, 0, 1, lattice, ws);
      if (with_self) {
        run_self_sos(form0, 0, lattice, ws);
        run_self_sos(form1, 1, lattice, ws);
      }
    }

    auto apply_to_form = [&form0, &form1](int tag, auto &&f) {
      if (tag == 0)
        f(form0);
      else
        f(form1);
    };
    finalize_identity(ws, apply_to_form, Index(2), with_self, !primitives);
  }

  /// N forms. Cross records for every pair; per-form self records when
  /// the mode carries `self_intersections`. A one-form range is the
  /// self-only build (the bit is implied — nothing else remains).
  template <typename Iterator, std::size_t N, typename Lattice>
  auto build(tf::range<Iterator, N> forms, const Lattice &lattice,
             tf::intersect_config config = {}) -> void {
    clear();
    take_vertex_offsets(lattice);

    const auto n = Index(forms.size());
    const bool primitives = config.mode & tf::intersect_mode::primitives;
    const bool with_self =
        n == 1 || (config.mode & tf::intersect_mode::self_intersections);

    tf::local_value<workspace_t> ws;
    tbb::task_group tg;
    for (Index i = 0; i < n; ++i)
      for (Index j = i + 1; j < n; ++j)
        tg.run([&, i, j]() {
          if (primitives)
            run_primitives_pair(forms[i], forms[j], int(i), int(j), lattice,
                                ws);
          else
            run_sos_pair(forms[i], forms[j], int(i), int(j), lattice, ws);
        });
    if (with_self)
      for (Index k = 0; k < n; ++k)
        tg.run([&, k]() {
          if (primitives)
            run_self_primitives(forms[k], int(k), lattice, ws);
          else
            run_self_sos(forms[k], int(k), lattice, ws);
        });
    tg.wait();

    auto apply_to_form = [forms](int tag, auto &&f) { f(forms[tag]); };
    finalize_identity(ws, apply_to_form, n, with_self, !primitives);
  }

  /// The face carrier: every face a point was delivered to, its own
  /// deliveries in one block. Block position is the face's identity
  /// through everything below.
  auto deliveries() const {
    return tf::make_offset_block_range(_delivery_offsets, _deliveries);
  }

  /// The deliveries belonging to one form.
  auto deliveries(Index tag) const {
    return tag_slice(_delivery_offsets, _deliveries, tag);
  }

  auto flat_deliveries() const { return tf::make_range(_deliveries); }

  /// The pair records at the carrier's block positions — empty where a
  /// face's pairs all proved inert.
  auto intersections() const {
    return tf::make_offset_block_range(_intersections_offsets, _intersections);
  }

  /// The groups belonging to one form.
  auto intersections(Index tag) const {
    return tag_slice(_intersections_offsets, _intersections, tag);
  }

  auto flat_intersections() const { return tf::make_range(_intersections); }

  auto tag_offsets() const { return tf::make_range(_tag_offsets); }

  auto get_flat_index(const intersection_t &rec) const -> Index {
    return Index(&rec - _intersections.begin());
  }

  /// Point ids below this are kind V; the rest are kind E.
  auto n_vertex_points() const -> Index {
    return Index(_identities.vertex_anchors.size());
  }

  /// The bound of the canonical point id space.
  auto n_points() const -> Index {
    return Index(_identities.vertex_anchors.size() +
                 _identities.home_edges.size());
  }

  /// The original vertex a kind-V point sits on: the lowest (tag,
  /// vertex) of its identified class.
  auto vertex_anchor(Index id) const
      -> const tf::intersect::vertex_anchor<Index> & {
    return _identities.vertex_anchors[std::size_t(id)];
  }

  /// The carrier a kind-E point sits on, as canonical flat vertex ids.
  auto home_edge(Index id) const -> const tf::intersect::home_edge<Index> & {
    return _identities
        .home_edges[std::size_t(id) - _identities.vertex_anchors.size()];
  }

  /// A kind-E point's exact position on its carrier, as a fraction of
  /// the carrier's span from `home_edge(id).u` to `home_edge(id).v`.
  /// The classifier that found the incidence stated it, from the same
  /// signs the classification is: rebuilding it from the point's
  /// generators reproduces it exactly, and nothing here does.
  auto exact_parameter(Index id) const
      -> const tf::exact::edge_parameter<Int> & {
    return _identities
        .parameters[std::size_t(id) - _identities.vertex_anchors.size()];
  }

  /// Per carrier, all canonical points incident to it in ascending
  /// exact parameter — kind-E classes and the vertices that lie on the
  /// carrier alike. Blocks align with @ref edge_carriers.
  auto edge_splits() const {
    return tf::make_offset_block_range(_identities.edge_splits.offsets_buffer(),
                                       _identities.edge_splits.data_buffer());
  }

  /// The carriers, ascending — binary-searchable by canonical edge.
  auto edge_carriers() const {
    return tf::make_range(_identities.edge_carriers);
  }

  /// The canonical flat vertex id of an original vertex. Total: a
  /// vertex no record identified maps to itself.
  auto canonical_vertex(std::int16_t tag, Index vid) const -> Index {
    return tf::intersect::canonical_flat_vertex(
        _vertex_identifications, _vertex_offsets[std::size_t(tag)] + vid);
  }

  /// Per-form bases of the flat vertex space, plus its total.
  auto vertex_offsets() const { return tf::make_range(_vertex_offsets); }

  auto clear() -> void {
    _deliveries.clear();
    _delivery_offsets.clear();
    _intersections.clear();
    _intersections_offsets.clear();
    _tag_offsets.clear();
    _vertex_offsets.clear();
    _vertex_identifications.clear();
    _identities.clear();
  }

private:
  template <typename Offsets, typename Data>
  auto tag_slice(const Offsets &offsets, const Data &data, Index tag) const {
    auto begin = _tag_offsets[std::size_t(tag)];
    auto end = _tag_offsets[std::size_t(tag) + 1];
    return tf::make_offset_block_range(
        tf::make_range(offsets.begin() + begin, offsets.begin() + end + 1),
        data);
  }

  template <typename Policy> static constexpr auto assert_form_policies() {
    static_assert(tf::has_tree_policy<Policy>, "Use polygons | tf::tag(tree)");
    static_assert(tf::has_manifold_edge_link_policy<Policy>,
                  "Use polygons | tf::tag(manifold_edge_link)");
    static_assert(tf::has_face_membership_policy<Policy>,
                  "Use polygons | tf::tag(face_membership)");
  }

  template <typename Policy0, typename Policy1, typename Lattice>
  auto run_sos_pair(const tf::polygons<Policy0> &form0,
                    const tf::polygons<Policy1> &form1, int tag0, int tag1,
                    const Lattice &lattice, tf::local_value<workspace_t> &ws) {
    auto &&mel0 = form0.manifold_edge_link();
    auto &&mel1 = form1.manifold_edge_link();
    tf::intersect::search_face_pairs(
        form0, form1, tag0, tag1, lattice, ws,
        [&, tag0, tag1](workspace_t &w) {
          tf::intersect::sos_process(w, tag0, tag1, mel0, mel1);
        });
  }

  template <typename Policy0, typename Policy1, typename Lattice>
  auto run_primitives_pair(const tf::polygons<Policy0> &form0,
                           const tf::polygons<Policy1> &form1, int tag0,
                           int tag1, const Lattice &lattice,
                           tf::local_value<workspace_t> &ws) {
    auto &&mel0 = form0.manifold_edge_link();
    auto &&mel1 = form1.manifold_edge_link();
    auto &&fm0 = form0.face_membership();
    auto &&fm1 = form1.face_membership();
    tf::intersect::search_face_pairs(
        form0, form1, tag0, tag1, lattice, ws, [&, tag0, tag1](workspace_t &w) {
          tf::intersect::primitives_process(w, form0, form1, tag0, tag1, mel0,
                                            mel1, fm0, fm1);
        });
  }

  template <typename Policy, typename Lattice>
  auto run_self_sos(const tf::polygons<Policy> &form, int tag,
                    const Lattice &lattice, tf::local_value<workspace_t> &ws) {
    auto &&mel = form.manifold_edge_link();
    tf::intersect::search_face_pairs_self(
        form, tag, lattice, ws, [&, tag](workspace_t &w, bool is_self) {
          tf::intersect::self_sos_process(w, is_self, form, tag, mel);
        });
  }

  template <typename Policy, typename Lattice>
  auto run_self_primitives(const tf::polygons<Policy> &form, int tag,
                           const Lattice &lattice,
                           tf::local_value<workspace_t> &ws) {
    auto &&mel = form.manifold_edge_link();
    auto &&fm = form.face_membership();
    tf::intersect::search_face_pairs_self(
        form, tag, lattice, ws, [&, tag](workspace_t &w, bool is_self) {
          tf::intersect::self_process(w, is_self, form, tag, mel, fm);
        });
  }

  /// A record's copies are a function of it, of the topology and of the
  /// gate, so the fan's size is a count and never a growth: the same walk
  /// states how many there are and then writes them — counts, one prefix
  /// per currency, one allocation each, and disjoint slices. Appending
  /// instead re-copies the whole table on every growth, which on a tiling
  /// model is the table's own size again and again.
  template <typename Count, typename Allocate, typename Write>
  static auto write_fans(std::array<tf::buffer<std::size_t>, 2> &offsets,
                         std::size_t n_raw, const Count &count,
                         const Allocate &allocate, const Write &write)
      -> void {
    for (auto &side : offsets) {
      side.allocate(n_raw + 1);
      side[0] = 0;
    }
    tf::parallel_for_each(
        tf::make_sequence_range(n_raw),
        [&](std::size_t r, fan_scratch_t &scratch) {
          const auto counts = count(r, scratch);
          offsets[0][r + 1] = counts[0];
          offsets[1][r + 1] = counts[1];
        },
        fan_scratch_t{}, tf::checked);
    for (std::size_t r = 1; r <= n_raw; ++r) {
      offsets[0][r] += offsets[0][r - 1];
      offsets[1][r] += offsets[1][r - 1];
    }
    allocate(offsets[0][n_raw], offsets[1][n_raw]);
    tf::parallel_for_each(
        tf::make_sequence_range(n_raw),
        [&](std::size_t r, fan_scratch_t &scratch) {
          write(r, offsets[0][r], offsets[1][r], scratch);
        },
        fan_scratch_t{}, tf::checked);
  }

  /// One path for both record families. Records collapse on their
  /// generators, identity is settled on that pre-duplication set — a
  /// record is its fact there — and the fan propagates the canonical
  /// names: a copy inherits its parent's point id. Only the duplicator's
  /// shared-vertex deliveries arrive unnamed, as sentinels past the
  /// fact-slot count, and resolve after the fan.
  ///
  /// The fan states two facts and they have two carriers. A contact
  /// names a canonical point at a feature of a face, and it proves a pair
  /// of faces meet there. A pair record already names the point at each
  /// of its two faces, so the identity currency is only the difference —
  /// the faces no admitted pair names — and a pole whose fans are not
  /// products states none of it and pays for none of it.
  ///
  /// The gate reads what a face was delivered, which the pair currency
  /// cannot state before it runs, so the gate's own table comes from a
  /// ticket pass: `{flat face, point}`, eight bytes, and only for the
  /// faces some product fan touches. Where nothing is a product there are
  /// no marks, no tickets, and no second sort.
  template <typename ApplyToForm>
  auto finalize_identity(tf::local_value<workspace_t> &ws,
                         const ApplyToForm &apply_to_form, Index n_tags,
                         bool with_self, bool sos) -> void {
    auto &raw = _raw;
    auto &parameters = _parameters;
    raw.clear();
    parameters.clear();
    // The coplanar pair fact is discovered in one kernel call while the
    // pair's contacts may be emitted through other calls' representative
    // gating, so the fact is joined onto the records after the fan. That
    // join is all this table is: it is built here, read by the fan's gate
    // and by distribute_coplanar_flags, and released — the per-record bit
    // is the only coplanar carrier anything downstream reads.
    tf::buffer<std::array<Index, 4>> coplanar_pairs;
    tf::intersect::merge_face_pair_workspaces(ws, raw, parameters,
                                              coplanar_pairs);
    tf::intersect::normalize_coplanar_pairs(coplanar_pairs);

    // The primitives cross kernels are representative-gated — one raw
    // record per fact, by proof — so only the self family (and the
    // untraced SoS path) can restate one. A missed duplicate could only
    // waste fan work, never change an id: formation is idempotent and
    // the post-fan collapse always runs.
    if (with_self || sos)
      tf::intersect::dedup_generator_records(raw);
    tf::intersect::identify_vertices(raw, parameters,
                                     _vertex_identifications);
    if (raw.size() == 0)
      return tf::intersect::group_intersection_carriers(
          _intersections, _deliveries, n_tags, _intersections_offsets,
          _delivery_offsets, _tag_offsets);
    tf::intersect::form_point_identities<Index, Int>(
        raw, _vertex_offsets, _vertex_identifications, parameters, _scratch,
        _identities, _with_edge_splits);

    const auto sentinel_base =
        (!with_self || sos) ? Index(0) : Index(parameters.size());
    const auto self_sentinel = [&](const intersection_t &rec) {
      return sentinel_base == Index(0)
                 ? Index(0)
                 : sentinel_base + _vertex_offsets[std::size_t(rec.tag)];
    };
    tf::buffer<Index> face_offsets;
    face_offsets.push_back(Index(0));
    for (Index t = 0; t < n_tags; ++t)
      apply_to_form(int(t), [&](const auto &form) {
        face_offsets.push_back(face_offsets[std::size_t(t)] +
                               Index(form.faces().size()));
      });
    const auto flat_face = [&](std::int16_t tag, Index object) {
      return face_offsets[std::size_t(tag)] + object;
    };
    // A fan whose extents are not a product states no more pairs than
    // features, so it asks the gate nothing and needs no mark. Reading
    // the two extents is what the mark costs on a pole with no products
    // to prune.
    const auto is_product = [&](const intersection_t &rec) {
      bool answer = false;
      apply_to_form(int(rec.tag), [&](const auto &form0) {
        apply_to_form(int(rec.tag_other), [&](const auto &form1) {
          answer = tf::intersect::intersection_sides_are_a_product<Index>(
              form0.faces(), form0.face_membership(),
              form0.manifold_edge_link(), form1.faces(),
              form1.face_membership(), form1.manifold_edge_link(), rec);
        });
      });
      return answer;
    };
    const auto expand = [&](const intersection_t &rec, fan_scratch_t &scratch) {
      bool prunable = false;
      apply_to_form(int(rec.tag), [&](const auto &form0) {
        apply_to_form(int(rec.tag_other), [&](const auto &form1) {
          prunable = tf::intersect::expand_intersection_sides<Index>(
              form0.faces(), form0.face_membership(),
              form0.manifold_edge_link(), form1.faces(),
              form1.face_membership(), form1.manifold_edge_link(), rec,
              rec.tag == rec.tag_other, scratch.neighbors, scratch.own,
              scratch.other);
        });
      });
      return prunable;
    };

    // The gate's whole tier hangs off one question, asked first and
    // asked cheaply: does any fan state more pairs than features? The
    // extents answer it without materializing a fan, and a pole that
    // says no allocates nothing below this line — no mask, no tickets,
    // no delivered blocks, and a fan that asks the gate nothing.
    std::size_t n_products = 0;
    tf::blocked_reduce(
        tf::make_sequence_range(raw.size()), n_products, std::size_t(0),
        [&](auto &&range, std::size_t &local) {
          for (const auto r : range)
            local += is_product(raw[r]) ? 1 : 0;
        },
        [](std::size_t local, std::size_t &total) { total += local; });

    tf::buffer<char> gated_faces;
    tf::buffer<std::array<Index, 2>> tickets;
    if (n_products != 0) {
      // A face a product fan touches has its whole point set ticketed,
      // by every record, or the gate could under-count and drop a chord.
      gated_faces.allocate(std::size_t(face_offsets[std::size_t(n_tags)]));
      std::fill(gated_faces.begin(), gated_faces.end(), char(0));
      tf::parallel_for_each(
          tf::make_sequence_range(raw.size()),
          [&](std::size_t r, fan_scratch_t &scratch) {
            if (!is_product(raw[r]) || !expand(raw[r], scratch))
              return;
            for (const auto &a : scratch.own)
              gated_faces[std::size_t(flat_face(raw[r].tag, a.object))] =
                  char(1);
            for (const auto &b : scratch.other)
              gated_faces[std::size_t(flat_face(raw[r].tag_other, b.object))] =
                  char(1);
          },
          fan_scratch_t{}, tf::checked);

      tf::buffer<std::size_t> at;
      at.allocate(raw.size() + 1);
      at[0] = 0;
      const auto ticket_count = [&](std::size_t r, fan_scratch_t &scratch) {
        expand(raw[r], scratch);
        std::size_t n = 0;
        for (const auto &a : scratch.own)
          n += gated_faces[std::size_t(flat_face(raw[r].tag, a.object))];
        for (const auto &b : scratch.other)
          n += gated_faces[std::size_t(flat_face(raw[r].tag_other, b.object))];
        return n;
      };
      tf::parallel_for_each(
          tf::make_sequence_range(raw.size()),
          [&](std::size_t r, fan_scratch_t &scratch) {
            at[r + 1] = ticket_count(r, scratch);
          },
          fan_scratch_t{}, tf::checked);
      for (std::size_t r = 1; r <= raw.size(); ++r)
        at[r] += at[r - 1];
      tickets.allocate(at[raw.size()]);
      tf::parallel_for_each(
          tf::make_sequence_range(raw.size()),
          [&](std::size_t r, fan_scratch_t &scratch) {
            expand(raw[r], scratch);
            auto *out = tickets.begin() + std::ptrdiff_t(at[r]);
            for (const auto &a : scratch.own) {
              const auto flat = flat_face(raw[r].tag, a.object);
              if (gated_faces[std::size_t(flat)])
                *out++ = {flat, raw[r].id};
            }
            for (const auto &b : scratch.other) {
              const auto flat = flat_face(raw[r].tag_other, b.object);
              if (gated_faces[std::size_t(flat)])
                *out++ = {flat, raw[r].id};
            }
          },
          fan_scratch_t{}, tf::checked);
    }

    tf::buffer<Index> delivered_offsets;
    tf::buffer<Index> delivered_points;
    tf::buffer<Index> coplanar_offsets;
    tf::buffer<Index> coplanar_partners;
    const auto gate = tf::intersect::make_pair_group_gate(
        tickets, coplanar_pairs, face_offsets, delivered_offsets,
        delivered_points, coplanar_offsets, coplanar_partners);

    const auto duplicate = [&](const intersection_t &rec, auto &sink,
                               auto &deliveries, fan_scratch_t &scratch) {
      if (rec.tag == rec.tag_other) {
        apply_to_form(int(rec.tag), [&](const auto &form) {
          tf::intersect::duplicate_intersection_self(
              form.faces(), form.face_membership(), form.manifold_edge_link(),
              rec, sink, deliveries, self_sentinel(rec), scratch);
        });
      } else {
        apply_to_form(int(rec.tag), [&](const auto &form0) {
          apply_to_form(int(rec.tag_other), [&](const auto &form1) {
            tf::intersect::duplicate_intersection(
                form0.faces(), form0.face_membership(),
                form0.manifold_edge_link(), form1.faces(),
                form1.face_membership(), form1.manifold_edge_link(), rec, sink,
                deliveries, gate, scratch);
          });
        });
      }
    };
    std::array<tf::buffer<std::size_t>, 2> fan_offsets;
    write_fans(
        fan_offsets, raw.size(),
        [&](std::size_t r, fan_scratch_t &scratch) {
          tf::intersect::duplicate_counting_sink<Index> pairs;
          tf::intersect::delivery_counting_sink<Index> deliveries;
          duplicate(raw[r], pairs, deliveries, scratch);
          return std::array<std::size_t, 2>{pairs.count, deliveries.count};
        },
        [&](std::size_t n_pairs, std::size_t n_deliveries) {
          _intersections.allocate(n_pairs);
          _deliveries.allocate(n_deliveries);
        },
        [&](std::size_t r, std::size_t pair_at, std::size_t delivery_at,
            fan_scratch_t &scratch) {
          tf::intersect::duplicate_span_sink<Index> pairs{
              _intersections.begin() + std::ptrdiff_t(pair_at)};
          tf::intersect::delivery_span_sink<Index> deliveries{
              _deliveries.begin() + std::ptrdiff_t(delivery_at)};
          duplicate(raw[r], pairs, deliveries, scratch);
        });

    tf::intersect::dedup_generator_records(_intersections);
    tf::intersect::dedup_point_deliveries(_deliveries);
    if (sentinel_base != Index(0))
      tf::intersect::resolve_delivered_vertices<Index, Int>(
          _intersections, _deliveries, sentinel_base, _vertex_offsets,
          _vertex_identifications, _identities);
    tf::intersect::distribute_coplanar_flags(_intersections, coplanar_offsets,
                                             coplanar_partners, face_offsets);
    tf::intersect::group_intersection_carriers(
        _intersections, _deliveries, n_tags, _intersections_offsets,
        _delivery_offsets, _tag_offsets);
  }

  /// The flat vertex prefix is @ref tf::exact::input_lattice's — it names the
  /// space the placed table is indexed by — so this class copies it and
  /// never restates it.
  template <typename Lattice> auto take_vertex_offsets(const Lattice &lattice)
      -> void {
    const auto &offsets = lattice.vertex_offsets();
    _vertex_offsets.allocate(offsets.size());
    std::copy(offsets.begin(), offsets.end(), _vertex_offsets.begin());
  }

  bool _with_edge_splits = true;
  tf::buffer<tf::intersect::point_delivery<Index>> _deliveries;
  tf::buffer<Index> _delivery_offsets;
  tf::buffer<intersection_t> _intersections;
  tf::buffer<Index> _intersections_offsets;
  tf::buffer<Index> _tag_offsets;
  tf::buffer<Index> _vertex_offsets;
  tf::buffer<std::array<Index, 2>> _vertex_identifications;
  tf::intersect::identity_scratch<Index, Int> _scratch;
  tf::buffer<intersection_t> _raw;
  tf::buffer<parameters_t> _parameters;
  tf::intersect::point_identities<Index, Int> _identities;
};

} // namespace tf
