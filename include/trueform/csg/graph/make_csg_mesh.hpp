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
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_copy_blocked.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../core/memory.hpp"
#include "../../core/none.hpp"
#include "../../core/point.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/slice.hpp"
#include "../../core/views/take.hpp"
#include "./make_csg_map_data.hpp"
#include "./make_csg_partition.hpp"
#include "./triangle_component_labels.hpp"
#include "tbb/task_group.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Build a CSG result mesh from the implicit N-form
///        arrangement and a precomputed `chosen_sides` selection.
///
/// Cut faces emit their exposed triangles; uncut faces keep their input
/// arity. The output
/// `polygons_buffer` type follows the input: an all-triangle input stays a
/// static `blocked<3>`; any other (quad / n-gon / mixed) input materialises
/// a dynamic-size buffer. Its faces are the pieces `chosen_sides`
/// selected, each wound as that array states.
///
/// `N_in` is the operands' static face arity, produced by the graph's
/// storage policy — a heterogeneous pair has no single forms element to
/// read it off.
template <typename RealOut, std::size_t N_in, bool WantLabels = false,
          typename Index, typename Arrangement, typename TagMask = tf::none_t>
auto make_csg_mesh(
    const Arrangement &arrangement,
    const tf::csg::graph::triangle_component_labels<Index> &labels,
    const tf::buffer<std::int8_t> &chosen_sides, const TagMask &tag_mask = {}) {
  const Index n_tags = arrangement.n_tags();
  auto apply_to_polygons = arrangement.apply_to_form();
  const auto &created_pts = arrangement.created_points();

  // ---- Stages 1 + 2: partition + vertex remap. ----------------------
  auto pids = make_csg_partition(arrangement, labels, chosen_sides, tag_mask);
  auto map_data = make_csg_map_data<Index>(arrangement, pids,
                                           apply_to_polygons);

  // The stream is canonical (conform resolved every corner in place), so
  // mapping needs no resolution.
  auto map_vertex = [&](auto tag, const auto &v) {
    return map_data.map_vertex(tag, v);
  };

  // ---- Stage 3: gather selected triangles per (form, label). --------
  auto exposed_tris = arrangement.global().exposed_tris();
  auto exposed_descriptors = arrangement.global().exposed_descriptors();
  auto triangle_slots = arrangement.triangle_slots();
  auto tag_offsets = arrangement.global().tag_offsets();

  // tri_data[t][L], tri_origins[t][L]: triangle stream per (form, label).
  // L = 0 reverse, L = 1 forward.
  tf::core::std_vector<std::array<tf::buffer<Index>, 2>> tri_data(n_tags);
  tf::core::std_vector<std::array<tf::buffer<Index>, 2>> tri_origins(n_tags);

  {
    tbb::task_group tg;
    for (Index t = 0; t < n_tags; ++t) {
      for (Index L = 0; L < 2; ++L) {
        tg.run([&, t, L] {
          auto &tri_buf = tri_data[t][L];
          auto &origin_buf = tri_origins[t][L];
          const Index lo = tag_offsets[t];
          for (auto lid : pids[t].cut_faces[L]) {
            const auto &tr = exposed_tris[lo + lid];
            tri_buf.push_back(map_vertex(t, tr[0]));
            tri_buf.push_back(map_vertex(t, tr[1]));
            tri_buf.push_back(map_vertex(t, tr[2]));
            origin_buf.push_back(
                exposed_descriptors[std::size_t(triangle_slots[lo + lid])]
                    .object);
          }
        });
      }
    }
    tg.wait();
  }

  // ---- Optional provenance: per output face, tag_labels (which input form)
  // and face_labels (original face id within that form). Built here from the
  // partition/triangulation results already in hand, in the SAME stream order
  // the mesh is assembled below (per form t: uncut fwd, uncut rev, tri fwd,
  // tri rev). Winding reversal on side-0 streams flips vertices within a face,
  // not the face order, so the labels ignore it. --------------------------
  tf::buffer<Index> tag_labels;
  tf::buffer<Index> face_labels;
  if constexpr (WantLabels) {
    Index total_faces = 0;
    for (Index t = 0; t < n_tags; ++t)
      total_faces += static_cast<Index>(pids[t].polygons[1].size()) +
                     static_cast<Index>(pids[t].polygons[0].size()) +
                     static_cast<Index>(tri_data[t][1].size() / 3) +
                     static_cast<Index>(tri_data[t][0].size() / 3);
    tag_labels.allocate(static_cast<std::size_t>(total_faces));
    face_labels.allocate(static_cast<std::size_t>(total_faces));
    Index off = 0;
    auto emit = [&](Index t, const auto &origins, Index size) {
      tf::parallel_fill(tf::take(tf::drop(tag_labels, off), size), t);
      tf::parallel_copy(origins, tf::take(tf::drop(face_labels, off), size));
      off += size;
    };
    for (Index t = 0; t < n_tags; ++t) {
      emit(t, pids[t].polygons[1],
           static_cast<Index>(pids[t].polygons[1].size()));
      emit(t, pids[t].polygons[0],
           static_cast<Index>(pids[t].polygons[0].size()));
      emit(t, tf::make_range(tri_origins[t][1]),
           static_cast<Index>(tri_data[t][1].size() / 3));
      emit(t, tf::make_range(tri_origins[t][0]),
           static_cast<Index>(tri_data[t][0].size() / 3));
    }
  }

  // ---- Stage 4: per-form remapped face range view. Welds never reach
  // here: a retired original's ring is promoted into the stream, so no
  // selected uncut face references anything the map does not know. ----
  auto original_maps = tf::make_offset_block_range(map_data.point_offsets,
                                                   map_data.original_map);
  // The uncut-face view's type follows the form's, so it is delivered to
  // a callback rather than returned — a heterogeneous pair has two such
  // types and no common one to return.
  auto with_uncut_faces = [&](Index t, Index L, const auto &fn) {
    apply_to_polygons(t, [&](const auto &form) {
      fn(tf::make_indirect_range(
          tf::make_range(pids[t].polygons[L]),
          tf::make_block_indirect_range(
              form.faces(),
              tf::make_mapped_range(
                  original_maps[t],
                  [off = map_data.original_offsets[t]](Index x) {
                    return x + off;
                  }))));
    });
  };

  // ---- Stage 6: points buffer (shared by both output-arity paths). --
  const Index total_pts =
      map_data.total_original_points + map_data.total_created_points;
  tf::points_buffer<RealOut, 3> pts_buf;
  pts_buf.allocate(total_pts);
  // the reader outlives every wait below: the emission's tasks read it
  const auto reader = arrangement.lattice().reader(apply_to_polygons);
  auto emit_points = [&](tbb::task_group &tg) {
    tf::arrangement::emit_arrangement_points<RealOut>(
        tg, n_tags, apply_to_polygons, map_data, created_pts, reader, pts_buf);
  };

  // ---- Stage 5: assemble output faces. Uncut faces keep their own arity;
  // only cut loops are triangles. Per form the four streams stay (uncut-fwd,
  // uncut-rev, tri-fwd, tri-rev); side-0 streams are reversed. The output
  // buffer type follows the input arity: all-triangle input stays a fast
  // `blocked<3>`; a non-triangle (or mixed) input materialises a dynamic
  // offset-block so each face keeps its vertex count. -----------------------

  // Stream descriptor: (forward src, size, reverse?, tag, uncut?) emitted in
  // output order. The one statement of that order — the uncut-face ranges
  // below are read off it rather than restating it.
  auto for_each_stream = [&](auto &&fn) {
    for (Index t = 0; t < n_tags; ++t) {
      with_uncut_faces(t, 1, [&](auto &&src) {
        fn(src, static_cast<Index>(pids[t].polygons[1].size()), false, t, true);
      });
      with_uncut_faces(t, 0, [&](auto &&src) {
        fn(src, static_cast<Index>(pids[t].polygons[0].size()), true, t, true);
      });
      fn(tf::make_blocked_range<3>(tf::make_range(tri_data[t][1])),
         static_cast<Index>(tri_data[t][1].size() / 3), false, t, false);
      fn(tf::make_blocked_range<3>(tf::make_range(tri_data[t][0])),
         static_cast<Index>(tri_data[t][0].size() / 3), true, t, false);
    }
  };

  // Per-tag [begin, end) of the faces emitted uncut. A tag's uncut
  // streams are adjacent, so one range per tag describes them.
  [[maybe_unused]] tf::blocked_buffer<Index, 2> uncut_faces;
  if constexpr (WantLabels) {
    uncut_faces.allocate(static_cast<std::size_t>(n_tags));
    for (Index t = 0; t < n_tags; ++t)
      uncut_faces[t][0] = uncut_faces[t][1] = Index(0);
    Index at = 0;
    Index seen = Index(-1);
    for_each_stream([&](auto &&, Index size, bool, Index t, bool uncut) {
      if (t != seen) {
        seen = t;
        uncut_faces[t][0] = uncut_faces[t][1] = at;
      }
      if (uncut)
        uncut_faces[t][1] = at + size;
      at += size;
    });
  }

  if constexpr (N_in == 3) {
    // Fast path: every face is a triangle, so the output is a `blocked<3>`.
    Index total_faces = 0;
    for_each_stream(
        [&](auto &&, Index size, bool, Index, bool) { total_faces += size; });
    tf::blocked_buffer<Index, 3> face_buf;
    face_buf.data_buffer().allocate(3 * static_cast<std::size_t>(total_faces));
    {
      tbb::task_group tg;
      Index offset = 0;
      for_each_stream([&](auto &&src, Index size, bool reverse, Index, bool) {
        auto dst = tf::slice(face_buf, offset, offset + size);
        if (reverse)
          tg.run([src, dst] { tf::parallel_copy_blocked_reverse(src, dst); });
        else
          tg.run([src, dst] { tf::parallel_copy(src, dst); });
        offset += size;
      });
      emit_points(tg);
      tg.wait();
    }
    auto mesh = tf::make_polygons_buffer(std::move(face_buf), std::move(pts_buf));
    if constexpr (WantLabels)
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels), std::move(uncut_faces),
                             std::move(map_data));
    else
      return mesh;
  } else {
    // General path: uncut faces keep their arity, cut loops are triangles, so
    // the output is a dynamic offset-block. Build per-face offsets in stream
    // order, then copy each stream into its face slice (reversed for side 0).
    Index n_faces = 0;
    for_each_stream(
        [&](auto &&, Index size, bool, Index, bool) { n_faces += size; });
    tf::buffer<Index> offs;
    offs.allocate(static_cast<std::size_t>(n_faces) + 1);
    offs[0] = 0;
    {
      // Write each face's vertex count into its offset slot, per stream in
      // parallel (streams own disjoint face ranges, so no contention); a
      // serial prefix sum then turns the counts into offsets.
      tbb::task_group tg;
      Index fbase = 0;
      for_each_stream([&](auto &&src, Index size, bool, Index, bool) {
        tg.run([src, fbase, &offs] {
          Index i = fbase;
          for (auto &&face : src)
            offs[++i] = static_cast<Index>(face.size());
        });
        fbase += size;
      });
      tg.wait();
    }
    for (Index i = 0; i < n_faces; ++i)
      offs[i + 1] += offs[i];
    tf::offset_block_buffer<Index, Index> face_buf;
    face_buf.offsets_buffer() = std::move(offs);
    face_buf.data_buffer().allocate(
        static_cast<std::size_t>(face_buf.offsets_buffer().back()));
    {
      tbb::task_group tg;
      Index offset = 0;
      for_each_stream([&](auto &&src, Index size, bool reverse, Index, bool) {
        auto dst = tf::slice(face_buf, offset, offset + size);
        if (reverse)
          tg.run([src, dst] { tf::parallel_copy_blocked_reverse(src, dst); });
        else
          tg.run([src, dst] { tf::parallel_copy_blocked(src, dst); });
        offset += size;
      });
      emit_points(tg);
      tg.wait();
    }
    auto mesh = tf::make_polygons_buffer(std::move(face_buf), std::move(pts_buf));
    if constexpr (WantLabels)
      return std::make_tuple(std::move(mesh), std::move(tag_labels),
                             std::move(face_labels), std::move(uncut_faces),
                             std::move(map_data));
    else
      return mesh;
  }
}

} // namespace tf::csg::graph
