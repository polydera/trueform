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
#include "./reverse_side_labels.hpp"
#include "./make_csg_map_data.hpp"
#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_iota.hpp"
#include "../../core/buffer.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/memory.hpp"
#include "../../core/point.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/static_size.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/slice.hpp"
#include "../../csg/csg_domains_index_map.hpp"
#include "./compute_domain_partition.hpp"
#include "./make_csg_domain_partition.hpp"
#include "../../cut/construct/make_arrangement_point_inverse.hpp"
#include "../../cut/impl/region_triangulator.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../exact/vertex_converter.hpp"
#include "tbb/parallel_sort.h"
#include "tbb/task_group.h"
#include <array>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf::csg::graph {

/// @ingroup csg
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
/// `created_pts` is the graph's unified created-points table (intersection
/// points followed by refinement-added points); created vertex ids index it
/// directly.
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
template <typename OutputCoordinateType = tf::none_t, bool WantLabels = false,
          bool WantPointMap = false, typename FormsRange, typename Index,
          typename Int, typename RealType>
auto make_csg_domains(
    const tf::cut::component_labels<Index> &ag,
    const tf::face_regions<Index, Int> &fr,
    const tf::cut::region_triangulator<Index, Int> &rt,
    const tf::buffer<tf::point<Int, 3>> &created_pts, const FormsRange &forms,
    const domain_partition<Index> &part,
    const tf::exact::vertex_converter<Int, RealType, 3> &conv) {
  using InputReal = tf::coordinate_type<decltype(forms[0])>;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;
  // Output cell arity follows the input: all-triangle input keeps a fast
  // static blocked<3>; any other (quad / n-gon / mixed) input gives each cell
  // a dynamic-size face buffer, with only the cut loops triangulated.
  constexpr std::size_t N_in =
      tf::static_size_v<decltype(forms[0].faces()[0])>;
  constexpr std::size_t N_out = (N_in == 3) ? std::size_t(3) : tf::dynamic_size;
  using out_t = tf::polygons_buffer<Index, RealOut, 3, N_out>;

  const Index n_tags = static_cast<Index>(forms.size());
  const Index n_kept = part.n_kept;
  auto apply_to_polygons = [&](Index t, const auto &f) { f(forms[t]); };

  // ---- Stages 1 + 2: domain partition + global vertex remap. --------
  // `pids` is consumed only by the map-data build (it decides which faces /
  // loops contribute vertices to the global space); the soup below keys on
  // `part.side_label` directly, not on the per-label lists.
  auto pids = make_csg_domain_partition(ag, rt, forms, part);
  const Index n_created = static_cast<Index>(created_pts.size());
  auto map_data = make_csg_map_data(rt, n_created, pids, apply_to_polygons);

  auto map_vertex = [&](auto tag, const auto &v) {
    return map_data.map_key(rt.resolve_key(tag, v));
  };

  // ---- Stage 4: global points buffer (same layout as make_csg_mesh). -
  const Index total_pts =
      map_data.total_original_points + map_data.total_created_points;
  tf::points_buffer<RealOut, 3> pts_buf;
  pts_buf.allocate(total_pts);
  {
    tbb::task_group tg;
    auto pts_range =
        tf::make_offset_block_range(map_data.original_offsets, pts_buf);
    for (Index t = 0; t < n_tags; ++t) {
      tg.run([&, t] {
        auto frame = tf::frame_of(forms[t]);
        if constexpr (std::is_integral_v<RealOut>) {
          tf::parallel_copy(
              tf::make_points(tf::make_indirect_range(
                  map_data.original_ids[t],
                  tf::make_mapped_range(
                      forms[t].points(),
                      [frame, &conv](auto pt) {
                        return conv.convert(tf::transformed(pt, frame));
                      }))),
              pts_range[t]);
        } else {
          tf::parallel_copy(
              tf::make_points(tf::make_indirect_range(
                  map_data.original_ids[t],
                  tf::make_mapped_range(
                      forms[t].points(),
                      [frame](auto pt) { return tf::transformed(pt, frame); }))),
              pts_range[t]);
        }
      });
    }
    tg.run([&] {
      if constexpr (std::is_integral_v<RealOut>) {
        tf::parallel_copy(
            tf::make_points(tf::make_mapped_range(
                map_data.created_ids,
                [&created_pts](Index id) {
                  return created_pts[std::size_t(id)];
                })),
            tf::drop(pts_buf, map_data.total_original_points));
      } else {
        tf::parallel_copy(
            tf::make_points(tf::make_mapped_range(
                map_data.created_ids,
                [&conv, &created_pts](Index id) {
                  return conv.deconvert(created_pts[std::size_t(id)]);
                })),
            tf::drop(pts_buf, map_data.total_original_points));
      }
    });
    tg.wait();
  }

  // Per-form remapped face stream (global vertex space), indexable by face id.
  tf::buffer<Index> canonical_original_map;
  const tf::buffer<Index> *emission_original_map = &map_data.original_map;
  if (rt.merges().size() != 0) {
    using vertex_t = tf::intersect::graph::vertex<Index>;
    using source_t = tf::intersect::graph::vertex_source;
    canonical_original_map.allocate(map_data.original_map.size());
    tf::parallel_for_each(
        tf::make_sequence_range(n_tags), [&](Index t) {
          const Index begin = map_data.point_offsets[t];
          const Index end = map_data.point_offsets[t + 1];
          tf::parallel_for_each(
              tf::make_sequence_range(begin, end),
              [&](Index flat) {
                canonical_original_map[flat] =
                    map_data.map_key(rt.resolve_key(
                        t, vertex_t{source_t::original, flat - begin,
                                    {0, tf::topo_type::face}})) -
                    map_data.original_offsets[t];
              },
              tf::checked);
        },
        tf::checked);
    emission_original_map = &canonical_original_map;
  }
  auto original_maps = tf::make_offset_block_range(
      map_data.point_offsets, *emission_original_map);
  auto all_mapped_faces = [&](Index t) {
    return tf::make_block_indirect_range(
        forms[t].faces(),
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
  using ag_t = tf::cut::component_labels<Index>;

  // Coincident faces always cut, so loops cover every stack (see
  // make_reverse_side_labels for the attribution rule).
  [[maybe_unused]] tf::buffer<std::array<Index, 2>> rev_label;
  if constexpr (WantLabels)
    rev_label = make_reverse_side_labels(ag.coplanar_pairs(), fr);
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
    tf::cut::make_arrangement_point_inverse(map_data, total_pts, gpt_tag,
                                            gpt_label);
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
        auto &tris = soup_t[t];
        auto &doms = dom_t[t];
        [[maybe_unused]] auto &tags = tag_t[t];
        [[maybe_unused]] auto &origs = orig_t[t];
        auto poly_labels = ag.polygon_labels(t);
        auto faces_t = all_mapped_faces(t);

        // Uncut surface faces (already triangles). A cut face carries
        // none_label — its sub-loops emit it instead — so it is skipped.
        // A promoted face keeps its surface label but swaps geometry:
        // its conforming triangle block emits instead of the plain
        // face (the refined path's conforming-stream rule).
        auto uncut_loops = rt.loops();
        for (Index f = 0; f < static_cast<Index>(poly_labels.size()); ++f) {
          const Index c = poly_labels[f];
          if (c == ag_t::none_label)
            continue;
          const Index d_fwd = side_label[2 * c + 1];
          const Index d_rev = side_label[2 * c + 0];
          if (d_fwd < 0 && d_rev < 0)
            continue;
          const Index pi = rt.promoted_index(t, f);
          if (pi >= 0) {
            const auto pr = rt.promoted_range(pi);
            for (Index li = pr[0]; li < pr[1]; ++li) {
              const auto &tr = uncut_loops[li];
              const Index g0 = map_vertex(t, tr[0]);
              const Index g1 = map_vertex(t, tr[1]);
              const Index g2 = map_vertex(t, tr[2]);
              if (d_fwd >= 0) {
                tris.push_back(std::array<Index, 3>{g0, g1, g2});
                doms.push_back(d_fwd);
                if constexpr (WantLabels) {
                  tags.push_back(t);
                  origs.push_back(f);
                }
              }
              if (d_rev >= 0) {
                tris.push_back(std::array<Index, 3>{g2, g1, g0});
                doms.push_back(d_rev);
                if constexpr (WantLabels) {
                  tags.push_back(t);
                  origs.push_back(f);
                }
              }
            }
            continue;
          }
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

        // Cut regions of this form, region-major: label / descriptor /
        // rev attribution are per REGION; each region emits its exposed
        // triangle block (`exposed_ranges`, blocks disjoint and in
        // stream order). Promoted faces emit above, under their
        // surface label — never here.
        auto region_labels = ag.loop_labels();
        auto region_descs = fr.descriptors();
        auto exposed = rt.exposed_ranges();
        auto loops = rt.loops();
        const Index rlo = fr.tag_offsets()[t];
        const Index rhi = fr.tag_offsets()[t + 1];
        for (Index rg = rlo; rg < rhi; ++rg) {
          const Index c = region_labels[rg];
          if (c == ag_t::none_label) // dead coplanar duplicate
            continue;
          const Index d_fwd = side_label[2 * c + 1];
          const Index d_rev = side_label[2 * c + 0];
          if (d_fwd < 0 && d_rev < 0)
            continue;
          const auto &desc = region_descs[rg];
          const auto r = exposed[rg];
          for (Index li = r[0]; li < r[1]; ++li) {
            const auto &tr = loops[li];
            const Index g0 = map_vertex(desc.tag, tr[0]);
            const Index g1 = map_vertex(desc.tag, tr[1]);
            const Index g2 = map_vertex(desc.tag, tr[2]);
            if (d_fwd >= 0) {
              tris.push_back(std::array<Index, 3>{g0, g1, g2});
              doms.push_back(d_fwd);
              if constexpr (WantLabels) {
                tags.push_back(desc.tag);
                origs.push_back(desc.object);
              }
            }
            if (d_rev >= 0) {
              tris.push_back(std::array<Index, 3>{g2, g1, g0});
              doms.push_back(d_rev);
              if constexpr (WantLabels) {
                tags.push_back(rev_label[std::size_t(rg)][0]);
                origs.push_back(rev_label[std::size_t(rg)][1]);
              }
            }
          }
        }
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
          auto poly_labels = ag.polygon_labels(t);
          auto faces_t = all_mapped_faces(t);
          auto uncut_loops = rt.loops();
          for (Index f = 0; f < static_cast<Index>(poly_labels.size()); ++f) {
            const Index c = poly_labels[f];
            if (c == ag_t::none_label)
              continue;
            const Index d_fwd = side_label[2 * c + 1];
            const Index d_rev = side_label[2 * c + 0];
            if (d_fwd < 0 && d_rev < 0)
              continue;
            const Index pi = rt.promoted_index(t, f);
            if (pi >= 0) {
              // promoted: the conforming triangle block replaces the
              // plain face, same surface label (refined-path rule)
              const auto pr = rt.promoted_range(pi);
              for (Index li = pr[0]; li < pr[1]; ++li) {
                const auto &tr = uncut_loops[li];
                const std::array<Index, 3> g{map_vertex(t, tr[0]),
                                             map_vertex(t, tr[1]),
                                             map_vertex(t, tr[2])};
                if (d_fwd >= 0)
                  push(tf::make_range(g), d_fwd, false, t, f);
                if (d_rev >= 0)
                  push(tf::make_range(g), d_rev, true, t, f);
              }
              continue;
            }
            auto gf = faces_t[f];
            if (d_fwd >= 0)
              push(gf, d_fwd, false, t, f);
            if (d_rev >= 0)
              push(gf, d_rev, true, t, f);
          }
          auto region_labels = ag.loop_labels();
          auto region_descs = fr.descriptors();
          auto exposed = rt.exposed_ranges();
          auto loops = rt.loops();
          const Index rlo = fr.tag_offsets()[t];
          const Index rhi = fr.tag_offsets()[t + 1];
          for (Index rg = rlo; rg < rhi; ++rg) {
            const Index c = region_labels[rg];
            if (c == ag_t::none_label)
              continue;
            const Index d_fwd = side_label[2 * c + 1];
            const Index d_rev = side_label[2 * c + 0];
            if (d_fwd < 0 && d_rev < 0)
              continue;
            const auto &desc = region_descs[rg];
            const auto r = exposed[rg];
            for (Index li = r[0]; li < r[1]; ++li) {
              const auto &tr = loops[li];
              const std::array<Index, 3> g{map_vertex(desc.tag, tr[0]),
                                           map_vertex(desc.tag, tr[1]),
                                           map_vertex(desc.tag, tr[2])};
              if (d_fwd >= 0)
                push(tf::make_range(g), d_fwd, false, desc.tag, desc.object);
              if (d_rev >= 0) {
                // rev_label is empty without WantLabels -- don't index it
                if constexpr (WantLabels)
                  push(tf::make_range(g), d_rev, true,
                       rev_label[std::size_t(rg)][0],
                       rev_label[std::size_t(rg)][1]);
                else
                  push(tf::make_range(g), d_rev, true, desc.tag, desc.object);
              }
            }
          }
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
