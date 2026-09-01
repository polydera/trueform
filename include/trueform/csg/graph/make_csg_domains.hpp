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
#include "../../arrangement/construct/emit_arrangement_points.hpp"
#include "../../arrangement/construct/make_arrangement_point_inverse.hpp"
#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_iota.hpp"
#include "../../core/buffer.hpp"
#include "../../core/memory.hpp"
#include "../../core/point.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/slice.hpp"
#include "../../csg/csg_domains_index_map.hpp"
#include "./compute_domain_partition.hpp"
#include "./make_csg_domain_partition.hpp"
#include "./make_csg_map_data.hpp"
#include "./reverse_side_labels.hpp"
#include "./triangle_component_labels.hpp"
#include "tbb/parallel_sort.h"
#include "tbb/task_group.h"
#include <array>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Extract every kept 3D domain of the implicit N-form
///        arrangement as its OWN watertight mesh.
///
/// Unlike @ref tf::csg::graph::make_csg_mesh (which merges one selected
/// region into a single boundary), this emits one `polygons_buffer` per
/// kept domain, keyed on the domain identity carried by `part.side_label`.
/// Two domains that share an inclusion bitvector (e.g. a sphere cut by a
/// plane) come out as two distinct meshes. Cut loops are the graph's
/// exposed triangle-grain loops — one triangle each; uncut faces keep
/// their input arity, so the cell type follows the input: all-triangle
/// input gives a fast static `blocked<3>`, any other input a dynamic-size
/// buffer.
///
/// Strategy: reuse the make_csg_mesh implicit-graph machinery (vertex
/// discovery, point materialisation) once over a `2 * n_kept`-label
/// partition, producing a global vertex space; build one flat
/// `{triangle, domain}` soup; then split it exactly like
/// @ref tf::split_into_domains — sort triangle ids by domain, bucket, and
/// re-deduplicate vertices per domain with a reused point_map + watermark.
///
/// @return `{ cells, ids }`: `cells[k]` is the mesh of dense domain `k`,
///         `ids[k]` the original (pre-dense) domain id.
/// `N_in` is the operands' static face arity, produced by the graph's
/// storage policy — a heterogeneous pair has no single forms element to
/// read it off.
template <typename RealOut, std::size_t N_in, bool WantLabels = false,
          bool WantPointMap = false, typename Index, typename Arrangement,
          typename Labels, typename TagMask = tf::none_t>
auto make_csg_domains(const Arrangement &arrangement, const Labels &labels,
                      const domain_partition<Index> &part,
                      const TagMask &tag_mask = {}) {
  const Index n_tags = arrangement.n_tags();
  auto apply_to_polygons = arrangement.apply_to_form();
  const auto &created_pts = arrangement.created_points();
  // Output cell arity follows the input: all-triangle input keeps a fast
  // static blocked<3>; any other (quad / n-gon / mixed) input gives each cell
  // a dynamic-size face buffer, with only the cut loops triangulated.
  constexpr std::size_t N_out = (N_in == 3) ? std::size_t(3) : tf::dynamic_size;
  using out_t = tf::polygons_buffer<Index, RealOut, 3, N_out>;

  const Index n_kept = part.n_kept;

  // ---- Stages 1 + 2: domain partition + global vertex remap. --------
  // `pids` is consumed only by the map-data build (it decides which faces /
  // loops contribute vertices to the global space); the soup below keys on
  // `part.side_label` directly, not on the per-label lists.
  auto pids = make_csg_domain_partition(arrangement, labels, part, tag_mask);
  auto map_data = make_csg_map_data<Index>(arrangement, pids,
                                           apply_to_polygons);

  // The stream is canonical (conform resolved every corner in place), so
  // mapping needs no resolution.
  auto map_vertex = [&](auto tag, const auto &v) {
    return map_data.map_vertex(tag, v);
  };

  // ---- Stage 4: global points buffer (same layout as make_csg_mesh). -
  const Index total_pts =
      map_data.total_original_points + map_data.total_created_points;
  tf::points_buffer<RealOut, 3> pts_buf;
  pts_buf.allocate(total_pts);
  {
    // the reader outlives the wait: the emission's tasks read it
    const auto reader = arrangement.lattice().reader(apply_to_polygons);
    tbb::task_group tg;
    tf::arrangement::emit_arrangement_points<RealOut>(
        tg, n_tags, apply_to_polygons, map_data, created_pts, reader, pts_buf);
    tg.wait();
  }

  // Per-form remapped face stream (global vertex space), indexable by face
  // id. Welds never reach here: a retired original's ring is promoted into
  // the stream, so no selected uncut face references anything the map does
  // not know.
  auto original_maps = tf::make_offset_block_range(map_data.point_offsets,
                                                   map_data.original_map);
  // Type follows the form, so the form is an argument — a heterogeneous
  // pair has two of these types and no common one.
  auto all_mapped_faces = [&](const auto &form, Index t) {
    return tf::make_block_indirect_range(
        form.faces(),
        tf::make_mapped_range(
            original_maps[t],
            [off = map_data.original_offsets[t]](Index x) { return x + off; }));
  };

  // ---- Stage 5a: build the output triangle soup (global vertex space,
  // winding baked) + per-triangle dense-domain label. Each surface face / cut
  // loop is visited ONCE. `side_label[2c+s]` is the dense kept domain the
  // component's side `s` bounds, or -1. Side 1 (forward) keeps the stored
  // winding; side 0 (reverse) flips it so the cell's normals face outward. A
  // cut loop's stored triangles go to both kept sides. (One task per form —
  // never per (form, domain).) ---------------------------------------------
  using labels_t = Labels;

  // Coincident faces always cut, so the stream covers every stack (see
  // make_reverse_side_labels for the attribution rule).
  [[maybe_unused]] tf::buffer<std::array<Index, 2>> rev_label;
  if constexpr (WantLabels)
    rev_label = make_reverse_side_labels<Index>(arrangement);

  auto exposed_tris = arrangement.global().exposed_tris();
  auto exposed_descriptors = arrangement.global().exposed_descriptors();
  auto triangle_slots = arrangement.triangle_slots();
  auto triangle_tags = arrangement.triangle_tags();
  auto triangle_labels = labels.triangle_labels();
  auto tag_offsets = arrangement.global().tag_offsets();
  const auto &side_label = part.side_label;
  tf::core::std_vector<out_t> cells(static_cast<std::size_t>(n_kept));

  // Optional per-cell provenance: tag_blocks[k][j] = the input form of cell
  // k's face j, face_blocks[k][j] = the original face id within that form.
  // Blocks run parallel to `cells`. Filled by build_provenance under
  // WantLabels, which gathers a per-soup-face (tag, origin) stream through the
  // same sorted permutation the cells are emitted in.
  tf::offset_block_buffer<Index, Index> tag_blocks;
  tf::offset_block_buffer<Index, Index> face_blocks;
  [[maybe_unused]] auto build_provenance =
      [&](const tf::buffer<Index> &tag_soup, const tf::buffer<Index> &orig_soup,
          const tf::buffer<Index> &perm, const tf::buffer<Index> &off) {
        const std::size_t n = perm.size();
        tag_blocks.data_buffer().allocate(n);
        face_blocks.data_buffer().allocate(n);
        for (std::size_t i = 0; i < n; ++i) {
          const Index p = perm[i];
          tag_blocks.data_buffer()[i] = tag_soup[static_cast<std::size_t>(p)];
          face_blocks.data_buffer()[i] = orig_soup[static_cast<std::size_t>(p)];
        }
        tag_blocks.offsets_buffer().allocate(off.size());
        face_blocks.offsets_buffer().allocate(off.size());
        for (std::size_t i = 0; i < off.size(); ++i) {
          tag_blocks.offsets_buffer()[i] = off[i];
          face_blocks.offsets_buffer()[i] = off[i];
        }
      };

  // Optional per-cell point provenance (WantPointMap). Build the global point
  // inverse once (the same one make_mesh_arrangement_index_map builds), then
  // the emit loops gather it through each cell's local->global point list as
  // the points are deduplicated -- no extra buffer, no second pass. `gpt_tag`
  // /`gpt_label`: global pts_buf point -> (form, input point); created carry
  // the end sentinels. Both block-data buffers grow in lockstep, so a single
  // `cell_pt_off` offset array serves both.
  tf::offset_block_buffer<Index, Index> point_tag_blocks;
  tf::offset_block_buffer<Index, Index> point_blocks;
  tf::buffer<Index> gpt_tag;
  tf::buffer<Index> gpt_label;
  tf::buffer<Index> cell_pt_off;
  if constexpr (WantPointMap) {
    tf::arrangement::make_arrangement_point_inverse(map_data, total_pts,
                                                    gpt_tag, gpt_label);
    cell_pt_off.reserve(static_cast<std::size_t>(n_kept) + 1);
    cell_pt_off.push_back(Index(0));
    point_tag_blocks.data_buffer().reserve(static_cast<std::size_t>(total_pts));
    point_blocks.data_buffer().reserve(static_cast<std::size_t>(total_pts));
  }

  if constexpr (N_in == 3) {
  tf::core::std_vector<tf::buffer<std::array<Index, 3>>> soup_t(n_tags);
  tf::core::std_vector<tf::buffer<Index>> dom_t(n_tags);
  tf::core::std_vector<tf::buffer<Index>> tag_t(n_tags);
  tf::core::std_vector<tf::buffer<Index>> orig_t(n_tags);
  {
    tbb::task_group tg;
    for (Index t = 0; t < n_tags; ++t)
      tg.run([&, t] {
        // The emission keys on side_label directly, not on the partition
        // lists — a masked-out form is skipped here too, or discovery and
        // emission would disagree.
        if constexpr (!std::is_same_v<TagMask, tf::none_t>)
          if (!tag_mask[std::size_t(t)])
            return;
        apply_to_polygons(t, [&](const auto &form) {
          auto &tris = soup_t[t];
          auto &doms = dom_t[t];
          [[maybe_unused]] auto &tags = tag_t[t];
          [[maybe_unused]] auto &origs = orig_t[t];
          auto poly_labels = labels.polygon_labels(t);
          auto faces_t = all_mapped_faces(form, t);

          // Uncut surface faces (already triangles). A cut face carries
          // none_label — its triangles emit it instead — so it is
          // skipped.
          for (Index f = 0; f < static_cast<Index>(poly_labels.size()); ++f) {
            const Index c = poly_labels[f];
            if (c == labels_t::none_label)
              continue;
            const Index d_fwd = side_label[2 * c + 1];
            const Index d_rev = side_label[2 * c + 0];
            if (d_fwd < 0 && d_rev < 0)
              continue;
            auto gf = faces_t[f];
            if (d_fwd >= 0) {
              tris.push_back(std::array<Index, 3>{gf[0], gf[1], gf[2]});
              doms.push_back(d_fwd);
              if constexpr (WantLabels) {
                tags.push_back(t);
                origs.push_back(f);
              }
            }
            if (d_rev >= 0) {
              tris.push_back(std::array<Index, 3>{gf[2], gf[1], gf[0]});
              doms.push_back(d_rev);
              if constexpr (WantLabels) {
                tags.push_back(t);
                origs.push_back(f);
              }
            }
          }

          // Cut faces of this form: their slice of the exposed triangle
          // stream. A dead coplanar duplicate carries none_label.
          for (Index e = tag_offsets[t]; e < tag_offsets[t + 1]; ++e) {
            const Index c = triangle_labels[e];
            if (c == labels_t::none_label)
              continue;
            const Index d_fwd = side_label[2 * c + 1];
            const Index d_rev = side_label[2 * c + 0];
            if (d_fwd < 0 && d_rev < 0)
              continue;
            const Index tag = triangle_tags[e];
            const auto &tr = exposed_tris[e];
            const Index g0 = map_vertex(tag, tr[0]);
            const Index g1 = map_vertex(tag, tr[1]);
            const Index g2 = map_vertex(tag, tr[2]);
            if (d_fwd >= 0) {
              tris.push_back(std::array<Index, 3>{g0, g1, g2});
              doms.push_back(d_fwd);
              if constexpr (WantLabels) {
                tags.push_back(tag);
                origs.push_back(
                    exposed_descriptors[std::size_t(triangle_slots[e])].object);
              }
            }
            if (d_rev >= 0) {
              tris.push_back(std::array<Index, 3>{g2, g1, g0});
              doms.push_back(d_rev);
              if constexpr (WantLabels) {
                tags.push_back(rev_label[std::size_t(e)][0]);
                origs.push_back(rev_label[std::size_t(e)][1]);
              }
            }
          }
        });
      });
    tg.wait();
  }

  // Concatenate the per-form soups into one flat (triangle, domain) stream.
  tf::buffer<Index> tag_off;
  tag_off.allocate(static_cast<std::size_t>(n_tags) + 1);
  tag_off[0] = 0;
  for (Index t = 0; t < n_tags; ++t)
    tag_off[t + 1] = tag_off[t] + static_cast<Index>(soup_t[t].size());
  const Index n_tris = tag_off[n_tags];

  tf::buffer<std::array<Index, 3>> soup;
  soup.allocate(static_cast<std::size_t>(n_tris));
  tf::buffer<Index> dom;
  dom.allocate(static_cast<std::size_t>(n_tris));
  tf::buffer<Index> tag_soup;
  tf::buffer<Index> orig_soup;
  if constexpr (WantLabels) {
    tag_soup.allocate(static_cast<std::size_t>(n_tris));
    orig_soup.allocate(static_cast<std::size_t>(n_tris));
  }
  {
    tbb::task_group tg;
    for (Index t = 0; t < n_tags; ++t)
      tg.run([&, t] {
        tf::parallel_copy(tf::make_range(soup_t[t]),
                          tf::slice(soup, tag_off[t], tag_off[t + 1]));
        tf::parallel_copy(tf::make_range(dom_t[t]),
                          tf::slice(dom, tag_off[t], tag_off[t + 1]));
        if constexpr (WantLabels) {
          tf::parallel_copy(tf::make_range(tag_t[t]),
                            tf::slice(tag_soup, tag_off[t], tag_off[t + 1]));
          tf::parallel_copy(tf::make_range(orig_t[t]),
                            tf::slice(orig_soup, tag_off[t], tag_off[t + 1]));
        }
      });
    tg.wait();
  }

  // ---- Stage 5b: split the soup exactly like tf::split_into_domains —
  // sort triangle ids by domain, bucket via compute_offsets, then emit
  // each domain with a reused point_map + watermark. -----------------------
  tf::buffer<Index> tri_ids;
  tri_ids.allocate(static_cast<std::size_t>(n_tris));
  tf::parallel_iota(tri_ids, Index(0));
  // Tie-break on triangle id so equal-domain order (and thus the per-cell
  // vertex numbering) is deterministic across runs.
  tbb::parallel_sort(tri_ids.begin(), tri_ids.end(), [&](Index a, Index b) {
    return dom[a] != dom[b] ? dom[a] < dom[b] : a < b;
  });
  tf::buffer<Index> offsets;
  offsets.reserve(static_cast<std::size_t>(n_kept));
  tf::compute_offsets(tri_ids, std::back_inserter(offsets), Index(0),
                      [&](Index a, Index b) { return dom[a] == dom[b]; });
  auto groups = tf::make_offset_block_range(offsets, tri_ids);

  tf::buffer<Index> point_map;
  point_map.allocate(static_cast<std::size_t>(
      map_data.total_original_points + map_data.total_created_points));
  tf::parallel_fill(point_map, Index(-1));
  Index point_sentinel = 0;
  tf::buffer<Index> pt_ids;

  for (auto &&group : groups) {
    if (group.size() == 0)
      continue;
    const Index k = dom[group.front()];
    auto &out = cells[static_cast<std::size_t>(k)];
    auto &data = out.faces_buffer().data_buffer();
    data.allocate(3 * group.size());

    Index current = point_sentinel;
    pt_ids.clear();
    Index write_at = 0;
    auto emit = [&](Index g) {
      auto &v = point_map[g];
      if (v == -1 || v < point_sentinel) {
        pt_ids.push_back(g);
        v = current++;
      }
      data[write_at++] = v - point_sentinel;
    };
    for (Index id : group) {
      const auto &tri = soup[id];
      emit(tri[0]);
      emit(tri[1]);
      emit(tri[2]);
    }

    out.points_buffer().allocate(pt_ids.size());
    tf::parallel_copy(tf::make_indirect_range(pt_ids, pts_buf.points()),
                      out.points());
    if constexpr (WantPointMap) {
      const Index base = static_cast<Index>(point_blocks.data_buffer().size());
      const std::size_t grown = static_cast<std::size_t>(base) + pt_ids.size();
      point_tag_blocks.data_buffer().reallocate(grown);
      point_blocks.data_buffer().reallocate(grown);
      tf::parallel_copy(tf::make_indirect_range(pt_ids, gpt_tag),
                        tf::drop(point_tag_blocks.data_buffer(), base));
      tf::parallel_copy(tf::make_indirect_range(pt_ids, gpt_label),
                        tf::drop(point_blocks.data_buffer(), base));
      cell_pt_off.push_back(static_cast<Index>(grown));
    }
    point_sentinel = current;
  }
  if constexpr (WantLabels)
    build_provenance(tag_soup, orig_soup, tri_ids, offsets);
  } else {
    // ---- Dynamic-arity path (non-triangle input). Build a (face, domain)
    // soup where uncut faces keep their arity and cut loops are triangles,
    // then split per domain into dynamic-size cells. The soup is built PER TAG
    // in parallel (offset-block faces + dom) and concatenated — same
    // parallelism as the triangle path. -------------------------------------
    tf::core::std_vector<tf::offset_block_buffer<Index, Index>> soup_t(n_tags);
    tf::core::std_vector<tf::buffer<Index>> dom_t(n_tags);
    tf::core::std_vector<tf::buffer<Index>> tag_t(n_tags);
    tf::core::std_vector<tf::buffer<Index>> orig_t(n_tags);
    {
      tbb::task_group tg;
      for (Index t = 0; t < n_tags; ++t)
        tg.run([&, t] {
          // Same rule as the triangle branch: emission keys on side_label,
          // so the mask must gate it alongside discovery.
          if constexpr (!std::is_same_v<TagMask, tf::none_t>)
            if (!tag_mask[std::size_t(t)])
              return;
          apply_to_polygons(t, [&](const auto &form) {
            auto &sp = soup_t[t];
            auto &dm = dom_t[t];
            [[maybe_unused]] auto &tags = tag_t[t];
            [[maybe_unused]] auto &origs = orig_t[t];
            tf::small_vector<Index, 16> tmp;
            auto push = [&](auto &&face, Index d, bool reverse, Index tag,
                            Index origin) {
              tmp.clear();
              const Index n = static_cast<Index>(face.size());
              for (Index i = 0; i < n; ++i)
                tmp.push_back(face[reverse ? n - 1 - i : i]);
              sp.push_back(tf::make_range(tmp));
              dm.push_back(d);
              if constexpr (WantLabels) {
                tags.push_back(tag);
                origs.push_back(origin);
              }
            };
            auto poly_labels = labels.polygon_labels(t);
            auto faces_t = all_mapped_faces(form, t);
            for (Index f = 0; f < static_cast<Index>(poly_labels.size()); ++f) {
              const Index c = poly_labels[f];
              if (c == labels_t::none_label)
                continue;
              const Index d_fwd = side_label[2 * c + 1];
              const Index d_rev = side_label[2 * c + 0];
              if (d_fwd < 0 && d_rev < 0)
                continue;
              auto gf = faces_t[f];
              if (d_fwd >= 0)
                push(gf, d_fwd, false, t, f);
              if (d_rev >= 0)
                push(gf, d_rev, true, t, f);
            }
            for (Index e = tag_offsets[t]; e < tag_offsets[t + 1]; ++e) {
              const Index c = triangle_labels[e];
              if (c == labels_t::none_label)
                continue;
              const Index d_fwd = side_label[2 * c + 1];
              const Index d_rev = side_label[2 * c + 0];
              if (d_fwd < 0 && d_rev < 0)
                continue;
              const Index tag = triangle_tags[e];
              const Index object =
                  exposed_descriptors[std::size_t(triangle_slots[e])].object;
              const auto &tr = exposed_tris[e];
              const std::array<Index, 3> g{map_vertex(tag, tr[0]),
                                           map_vertex(tag, tr[1]),
                                           map_vertex(tag, tr[2])};
              if (d_fwd >= 0)
                push(tf::make_range(g), d_fwd, false, tag, object);
              if (d_rev >= 0) {
                // rev_label is empty without WantLabels
                if constexpr (WantLabels)
                  push(tf::make_range(g), d_rev, true,
                       rev_label[std::size_t(e)][0],
                       rev_label[std::size_t(e)][1]);
                else
                  push(tf::make_range(g), d_rev, true, tag, object);
              }
            }
          });
        });
      tg.wait();
    }

    // Concatenate per-tag offset-block soups into one global soup + dom.
    tf::buffer<Index> face_off, data_off;
    face_off.allocate(static_cast<std::size_t>(n_tags) + 1);
    data_off.allocate(static_cast<std::size_t>(n_tags) + 1);
    face_off[0] = 0;
    data_off[0] = 0;
    for (Index t = 0; t < n_tags; ++t) {
      face_off[t + 1] = face_off[t] + static_cast<Index>(soup_t[t].size());
      data_off[t + 1] =
          data_off[t] + static_cast<Index>(soup_t[t].data_buffer().size());
    }
    const Index total_faces = face_off[n_tags];
    tf::offset_block_buffer<Index, Index> soup;
    soup.offsets_buffer().allocate(static_cast<std::size_t>(total_faces) + 1);
    soup.data_buffer().allocate(static_cast<std::size_t>(data_off[n_tags]));
    soup.offsets_buffer()[0] = 0;
    tf::buffer<Index> dom;
    dom.allocate(static_cast<std::size_t>(total_faces));
    tf::buffer<Index> tag_soup;
    tf::buffer<Index> orig_soup;
    if constexpr (WantLabels) {
      tag_soup.allocate(static_cast<std::size_t>(total_faces));
      orig_soup.allocate(static_cast<std::size_t>(total_faces));
    }
    {
      tbb::task_group tg;
      for (Index t = 0; t < n_tags; ++t)
        tg.run([&, t] {
          tf::parallel_copy(
              tf::make_range(soup_t[t].data_buffer()),
              tf::slice(soup.data_buffer(), data_off[t], data_off[t + 1]));
          tf::parallel_copy(tf::make_range(dom_t[t]),
                            tf::slice(dom, face_off[t], face_off[t + 1]));
          if constexpr (WantLabels) {
            tf::parallel_copy(tf::make_range(tag_t[t]),
                              tf::slice(tag_soup, face_off[t], face_off[t + 1]));
            tf::parallel_copy(
                tf::make_range(orig_t[t]),
                tf::slice(orig_soup, face_off[t], face_off[t + 1]));
          }
          const auto &lo = soup_t[t].offsets_buffer();
          const Index base = data_off[t], fbase = face_off[t];
          for (Index fi = 0; fi < static_cast<Index>(soup_t[t].size()); ++fi)
            soup.offsets_buffer()[fbase + fi + 1] = lo[fi + 1] + base;
        });
      tg.wait();
    }

    const Index n_faces = static_cast<Index>(dom.size());
    tf::buffer<Index> face_ids;
    face_ids.allocate(static_cast<std::size_t>(n_faces));
    tf::parallel_iota(face_ids, Index(0));
    tbb::parallel_sort(face_ids.begin(), face_ids.end(), [&](Index a, Index b) {
      return dom[a] != dom[b] ? dom[a] < dom[b] : a < b;
    });
    tf::buffer<Index> offsets;
    offsets.reserve(static_cast<std::size_t>(n_kept));
    tf::compute_offsets(face_ids, std::back_inserter(offsets), Index(0),
                        [&](Index a, Index b) { return dom[a] == dom[b]; });
    auto groups = tf::make_offset_block_range(offsets, face_ids);
    const auto &soup_off = soup.offsets_buffer();
    const auto &soup_dat = soup.data_buffer();

    tf::buffer<Index> point_map;
    point_map.allocate(static_cast<std::size_t>(total_pts));
    tf::parallel_fill(point_map, Index(-1));
    Index point_sentinel = 0;
    tf::buffer<Index> pt_ids;
    for (auto &&group : groups) {
      if (group.size() == 0)
        continue;
      const Index k = dom[group.front()];
      auto &out = cells[static_cast<std::size_t>(k)];
      auto &foff = out.faces_buffer().offsets_buffer();
      auto &fdata = out.faces_buffer().data_buffer();
      foff.allocate(group.size() + 1);
      foff[0] = 0;
      Index acc = 0, fi = 0;
      for (Index id : group) {
        acc += soup_off[id + 1] - soup_off[id];
        foff[++fi] = acc;
      }
      fdata.allocate(static_cast<std::size_t>(acc));
      Index current = point_sentinel, write_at = 0;
      pt_ids.clear();
      auto emit = [&](Index g) {
        auto &v = point_map[g];
        if (v == -1 || v < point_sentinel) {
          pt_ids.push_back(g);
          v = current++;
        }
        fdata[write_at++] = v - point_sentinel;
      };
      for (Index id : group)
        for (Index x = soup_off[id]; x < soup_off[id + 1]; ++x)
          emit(soup_dat[x]);
      out.points_buffer().allocate(pt_ids.size());
      tf::parallel_copy(tf::make_indirect_range(pt_ids, pts_buf.points()),
                        out.points());
      if constexpr (WantPointMap) {
        const Index base =
            static_cast<Index>(point_blocks.data_buffer().size());
        const std::size_t grown =
            static_cast<std::size_t>(base) + pt_ids.size();
        point_tag_blocks.data_buffer().reallocate(grown);
        point_blocks.data_buffer().reallocate(grown);
        tf::parallel_copy(tf::make_indirect_range(pt_ids, gpt_tag),
                          tf::drop(point_tag_blocks.data_buffer(), base));
        tf::parallel_copy(tf::make_indirect_range(pt_ids, gpt_label),
                          tf::drop(point_blocks.data_buffer(), base));
        cell_pt_off.push_back(static_cast<Index>(grown));
      }
      point_sentinel = current;
    }
    if constexpr (WantLabels)
      build_provenance(tag_soup, orig_soup, face_ids, offsets);
  }

  // ids[k] = original domain whose dense_of_domain == k.
  tf::buffer<Index> ids;
  ids.allocate(static_cast<std::size_t>(n_kept));
  for (Index d = 0; d < static_cast<Index>(part.dense_of_domain.size()); ++d) {
    const Index dense = part.dense_of_domain[d];
    if (dense >= 0)
      ids[dense] = d;
  }

  if constexpr (WantPointMap) {
    // Both block-data buffers grew in lockstep during emit, so cell_pt_off is
    // their shared offset layout: move it into one, copy it into the other.
    point_tag_blocks.offsets_buffer().allocate(cell_pt_off.size());
    tf::parallel_copy(tf::make_range(cell_pt_off),
                      tf::make_range(point_tag_blocks.offsets_buffer()));
    point_blocks.offsets_buffer() = std::move(cell_pt_off);

    tf::csg_domains_index_map<Index> imap;
    imap.face_tag_blocks = std::move(tag_blocks);
    imap.face_blocks = std::move(face_blocks);
    imap.point_tag_blocks = std::move(point_tag_blocks);
    imap.point_blocks = std::move(point_blocks);
    imap.n_original_points = map_data.total_original_points;
    imap.n_tags = n_tags;
    imap.n_output_points = total_pts;
    return std::make_tuple(std::move(cells), std::move(ids), std::move(imap));
  } else if constexpr (WantLabels) {
    return std::make_tuple(std::move(cells), std::move(ids),
                           std::move(tag_blocks), std::move(face_blocks));
  } else {
    return std::make_pair(std::move(cells), std::move(ids));
  }
}

} // namespace tf::csg::graph
