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
#include "../../core/array_hash.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/hash_set.hpp"
#include "../../core/range.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./label_plane_arrangement_components.hpp"
#include "./make_surface_component_labels.hpp"
#include "./resolve_face_edge.hpp"

#include "tbb/parallel_invoke.h"
#include "tbb/parallel_sort.h"
#include "tbb/task_group.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief The classification label tier: two tiers over disjoint
///        carriers, compacted into one dense component id space.
///
///   CUT CCL      over the arrangement's CELLS. A cell is bounded by
///                constraints and nothing else, so a triangulation's
///                interior edges are not merely filtered out of this
///                answer — they are not representable in it. The flood
///                crosses a piece iff the arrangement's fence says it
///                may; each triangle inherits the label of the cell it
///                came out of.
///   SURFACE CCL  per form, over the faces the arrangement never cut,
///                flood-filled through the prebuilt
///                `manifold_edge_link`. Uncut adjacency is already
///                solved there — joining identity triangles by vertex
///                identity would only pay for it a second time.
///   BRIDGES      cut <-> uncut across the source mesh's manifold edge:
///                a cut triangle slot on a piece no other live triangle
///                names, whose endpoints resolve to one original face
///                edge whose peer is uncut.
///
/// Dead (coplanar-duplicate) triangles carry `none_label` — their cell
/// is their survivor's, so the component loses nothing by it.
/// @ref triangle_labels covers the exposed stream (cut faces only),
/// @ref polygon_labels the uncut faces. The open mask is the manifold
/// edge link's boundary answer on both tiers.
template <typename Index> class triangle_component_labels {
public:
  using label_type = Index;
  static constexpr label_type none_label = label_type(-1);

  auto n_components() const -> Index { return _n_components; }
  /// @brief Per exposed triangle: its component; dead triangles
  ///        `none_label`.
  auto triangle_labels() const { return tf::make_range(_labels); }
  /// @brief Per-face surface labels for form `tag`: an uncut face
  ///        carries its component, a cut face `none_label` (its
  ///        triangles carry the labels).
  auto polygon_labels(Index tag) const {
    return tf::make_range(_polygon_labels[std::size_t(tag)]);
  }
  /// @brief Per-component open flag (`1` = touches an original mesh
  ///        boundary).
  auto open_component_mask() const {
    return tf::make_range(_open_component_mask);
  }

  /// @param arrangement   The arrangement graph: its exposed triangle
  ///                      stream, cells, piece incidence and fences,
  ///                      emission slots, tags, descriptors and dead mask.
  /// @param apply_to_form Callable `(Index tag, callback) -> void`
  ///        invoking `callback` with the tagged polygon form of `tag`
  ///        (each carrying a `manifold_edge_link`); not retained.
  template <typename Graph, typename ApplyToForm>
  auto build(const Graph &arrangement, const ApplyToForm &apply_to_form)
      -> void {
    const auto n_tags = arrangement.n_tags();

    _polygon_labels.resize(std::size_t(n_tags));
    tf::small_vector<Index, 4> n_surface_per_tag;
    n_surface_per_tag.resize(std::size_t(n_tags));
    Index K_cut = 0;
    tbb::parallel_invoke(
        [&] {
          _compute_surface_labels(arrangement, apply_to_form,
                                  n_surface_per_tag);
        },
        [&] { K_cut = _compute_cell_component_labels(arrangement); });
    auto form_offsets = _offset_surface_labels(n_surface_per_tag, K_cut);
    const Index total_nodes = form_offsets[form_offsets.size() - 1];

    auto bridges = _collect_bridges(arrangement, apply_to_form);
    _apply_bridges_and_compact(bridges, total_nodes);

    _compute_open_components(arrangement, apply_to_form);
  }

private:
  // tf::csg::graph::make_surface_component_labels reads only `.object` from
  // its descriptor entries
  struct cut_face_descriptor {
    Index object;
  };

  /// The cut tier's CCL is the arrangement's own: a flood over cells,
  /// crossing a piece iff its fence says the ground continues there. The
  /// stream only inherits the answer — a triangle takes the label of the
  /// cell it came out of, and a dead duplicate is masked because its
  /// survivor already carries their shared cell.
  template <typename Graph>
  auto _compute_cell_component_labels(const Graph &arrangement) -> Index {
    auto dead = arrangement.dead();
    auto rows = arrangement.row_of();
    const auto n_exp = Index(rows.size());

    auto cell_components =
        tf::csg::graph::label_plane_arrangement_components<Index, label_type>(
            arrangement.piece_incidence(),
            arrangement.piece_fences().crossable, arrangement.cells());

    _labels.allocate(std::size_t(n_exp));
    auto labels_of_row = tf::make_range(cell_components.labels);
    tf::parallel_for_each(
        tf::make_sequence_range(n_exp),
        [&](Index e) {
          _labels[std::size_t(e)] =
              dead[e] ? none_label
                      : labels_of_row[std::size_t(rows[std::size_t(e)])];
        },
        tf::checked);
    return cell_components.n_components;
  }

  /// Per-form surface CCL, in parallel; labels stay in each form's
  /// local id space until @ref _offset_surface_labels lifts them. The
  /// cut (and promoted) faces are masked off — their triangles are the
  /// cut tier's carriers.
  template <typename Graph, typename ApplyToForm>
  auto _compute_surface_labels(const Graph &arrangement,
                               const ApplyToForm &apply_to_form,
                               tf::small_vector<Index, 4> &n_surface_per_tag)
      -> void {
    const auto &ga = arrangement.global();
    auto descs = ga.exposed_descriptors();
    auto fob = ga.face_offset_base();
    const auto n_tags = arrangement.n_tags();

    tbb::task_group tg;
    for (Index t = 0; t < n_tags; ++t)
      tg.run([this, t, &apply_to_form, &descs, &fob, &n_surface_per_tag] {
        // exposure is tag-major with one slot per face and a base row
        // per tag in face_offset_base
        const auto base = Index(fob[std::size_t(t)] - t);
        const auto n_faces =
            Index(fob[std::size_t(t) + 1] - fob[std::size_t(t)] - 1);
        tf::buffer<cut_face_descriptor> cut_faces;
        tf::generic_generate(
            tf::make_sequence_range(n_faces), cut_faces,
            [&](Index object, tf::buffer<cut_face_descriptor> &out) {
              if (descs[base + object].plane != Index(-1))
                out.push_back({object});
            });
        tf::buffer<Index> no_deleted;
        apply_to_form(t, [&](const auto &form) {
          auto cl =
              tf::csg::graph::make_surface_component_labels<Index, label_type>(
                  form, tf::make_range(cut_faces), tf::make_range(no_deleted));
          n_surface_per_tag[std::size_t(t)] = cl.n_components;
          _polygon_labels[std::size_t(t)] = std::move(cl.labels);
        });
      });
    tg.wait();
  }

  /// Lift per-form surface labels into the global id space; returns
  /// per-form offsets, `form_offsets[n_tags]` = total node count.
  auto
  _offset_surface_labels(const tf::small_vector<Index, 4> &n_surface_per_tag,
                         Index K_cut) -> tf::buffer<Index> {
    const auto n_tags = Index(n_surface_per_tag.size());
    tf::buffer<Index> form_offsets;
    form_offsets.allocate(std::size_t(n_tags) + 1);
    form_offsets[0] = K_cut;
    for (Index t = 0; t < n_tags; ++t)
      form_offsets[std::size_t(t) + 1] =
          form_offsets[std::size_t(t)] + n_surface_per_tag[std::size_t(t)];

    tbb::task_group tg;
    for (Index t = 0; t < n_tags; ++t)
      tg.run([this, t, &form_offsets] {
        const Index offset = form_offsets[std::size_t(t)];
        tf::parallel_for_each(
            _polygon_labels[std::size_t(t)],
            [offset](auto &id) {
              if (id != none_label)
                id += offset;
            },
            tf::checked);
      });
    tg.wait();
    return form_offsets;
  }

  /// Bridge pairs between a cut triangle's component and the surface
  /// component of an uncut neighbour across the source mesh's manifold
  /// edge. A candidate slot lies on a piece no other live triangle names
  /// — the arrangement's own frontier, where the cut surface ends and the
  /// source mesh's link takes over.
  template <typename Graph, typename ApplyToForm>
  auto _collect_bridges(const Graph &arrangement,
                        const ApplyToForm &apply_to_form) const
      -> tf::buffer<std::array<Index, 2>> {
    tf::buffer<std::array<Index, 2>> bridges;

    const auto &ga = arrangement.global();
    auto tris = ga.exposed_tris();
    auto descs = ga.exposed_descriptors();
    auto slots = arrangement.triangle_slots();
    auto parents = ga.exposed_parent_of();
    auto dead = arrangement.dead();
    auto exposed_of_row = arrangement.exposed_of_row();
    const auto &incidence = arrangement.piece_incidence();
    const auto n_exp = Index(tris.size());

    auto is_frontier = [&](Index e, int s) {
      const auto piece = parents[std::size_t(e) * 3 + std::size_t(s)];
      if (piece == Index(-1))
        return false; // a filler never lies on an original side
      for (const auto row : incidence.rows_of_piece[std::size_t(piece)]) {
        const auto peer = exposed_of_row[std::size_t(row / Index(3))];
        if (peer == Index(-1) || peer == e || dead[peer])
          continue;
        return false;
      }
      return true;
    };

    tf::generic_generate(
        tf::make_sequence_range(n_exp), bridges,
        [&](Index e, auto &buffer, auto &set) {
          const auto cut_id = _labels[std::size_t(e)];
          if (cut_id == none_label)
            return;
          const auto &d = descs[slots[e]];
          apply_to_form(Index(d.tag), [&](const auto &form) {
            const auto &poly_labels_t = _polygon_labels[std::size_t(d.tag)];
            auto mel_t = form.manifold_edge_link();
            const auto face_size = form.faces()[d.object].size();
            const auto &tv = tris[e];
            for (int s = 0; s < 3; ++s) {
              if (!is_frontier(e, s))
                continue;
              const auto eidx = tf::csg::graph::resolve_face_edge(
                  tv[std::size_t(s)].sub_id,
                  tv[std::size_t((s + 1) % 3)].sub_id, face_size);
              if (!eidx)
                continue;
              auto m = mel_t[d.object][*eidx];
              if (!m.is_simple())
                continue;
              const Index peer = m.face_peer;
              const Index peer_label = poly_labels_t[std::size_t(peer)];
              if (peer_label == none_label)
                continue;
              const Index a = std::min(peer_label, Index(cut_id));
              const Index b = std::max(peer_label, Index(cut_id));
              std::array<Index, 2> pair{a, b};
              if (set.insert(pair).second)
                buffer.push_back(pair);
            }
          });
        },
            tf::hash_set<std::array<Index, 2>, tf::array_hash<Index, 2>>{});

    return bridges;
  }

  /// Collapse bridge pairs into components and remap every label
  /// buffer to the final dense ids.
  auto _apply_bridges_and_compact(tf::buffer<std::array<Index, 2>> &bridges,
                                  Index total_nodes) -> void {
    tbb::parallel_sort(bridges.begin(), bridges.end());
    bridges.erase_till_end(std::unique(bridges.begin(), bridges.end()));

    tf::buffer<Index> map;
    map.allocate(std::size_t(total_nodes));
    _n_components = tf::make_dense_equivalence_class_map(bridges, map);

    auto remap = [&map](auto &id) {
      if (id != none_label)
        id = map[id];
    };

    tbb::task_group tg;
    tg.run([&] { tf::parallel_for_each(_labels, remap, tf::checked); });
    for (std::size_t t = 0; t < _polygon_labels.size(); ++t)
      tg.run([&, t] {
        tf::parallel_for_each(_polygon_labels[t], remap, tf::checked);
      });
    tg.wait();
  }

  /// Mark every component touching a boundary edge as open. Writes are
  /// `1`-only, idempotent under data race.
  template <typename Graph, typename ApplyToForm>
  auto _compute_open_components(const Graph &arrangement,
                                const ApplyToForm &apply_to_form) -> void {
    _open_component_mask.allocate(std::size_t(_n_components));
    tf::parallel_fill(_open_component_mask, char(0));

    if (_n_components == 0)
      return;

    const auto &ga = arrangement.global();
    auto tris = ga.exposed_tris();
    auto descs = ga.exposed_descriptors();
    auto slots = arrangement.triangle_slots();
    const auto n_exp = Index(tris.size());
    const auto n_tags = arrangement.n_tags();

    tbb::task_group tg;
    tg.run([this, tris, descs, slots, n_exp, &apply_to_form] {
      tf::parallel_for_each(tf::make_sequence_range(n_exp), [&](Index e) {
        const auto c = _labels[std::size_t(e)];
        if (c == none_label)
          return;
        const auto &d = descs[slots[e]];
        apply_to_form(Index(d.tag), [&](const auto &form) {
          auto mel_t = form.manifold_edge_link();
          const auto face_size = form.faces()[d.object].size();
          const auto &tv = tris[e];
          for (int s = 0; s < 3; ++s) {
            const auto eidx = tf::csg::graph::resolve_face_edge(
                tv[std::size_t(s)].sub_id, tv[std::size_t((s + 1) % 3)].sub_id,
                face_size);
            if (!eidx)
              continue;
            if (mel_t[d.object][*eidx].is_boundary())
              _open_component_mask[std::size_t(c)] = char(1);
          }
        });
      });
    });

    for (Index t = Index(0); t < n_tags; ++t)
      tg.run([this, t, &apply_to_form] {
        apply_to_form(t, [&](const auto &form) {
          auto poly_labels_t = tf::make_range(_polygon_labels[std::size_t(t)]);
          auto mel_t = form.manifold_edge_link();
          tf::parallel_for_each(
              tf::enumerate(poly_labels_t), [this, mel_t](auto pair) {
                auto &&[f, label] = pair;
                if (label == none_label)
                  return;
                for (const auto &m : mel_t[Index(f)])
                  if (m.is_boundary()) {
                    _open_component_mask[std::size_t(label)] = char(1);
                    return;
                  }
              });
        });
      });
    tg.wait();
  }

  tf::buffer<label_type> _labels;
  tf::small_vector<tf::buffer<label_type>, 4> _polygon_labels;
  tf::buffer<char> _open_component_mask;
  Index _n_components = 0;
};

} // namespace tf::csg::graph
