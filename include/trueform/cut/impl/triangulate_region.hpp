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

#include "../../core/buffer.hpp"
#include "../../core/edges.hpp"
#include "../../core/point.hpp"
#include "../../core/points.hpp"
#include "../../core/range.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../core/views/constant.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/projection_axes.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../topology/topo_id.hpp"
#include "../detail/region_triangulation_identity.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "../face_regions.hpp"
#include "../detail/resolve_face_edge.hpp"
#include "./identify_resolved_crossings.hpp"
#include "./region_split_point.hpp"
#include "./region_triangulation_point.hpp"
#include "./resolve_region_vertex.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::cut {

template <typename Index, typename Int, typename Descriptor, typename Loop,
          typename HoleIds, typename ApplyToFace, typename GetMeshPoint,
          typename IntersectionPoints,
          typename InteriorPoints = tf::range<
              const tf::cut::detail::region_interior_point<Index, Int> *,
              tf::dynamic_size>>
auto triangulate_region(
    const tf::face_regions<Index, Int> &regions, const Descriptor &descriptor,
    const Loop &loop, const HoleIds &hole_ids,
    const ApplyToFace &apply_to_face, const GetMeshPoint &get_mesh_point,
    const IntersectionPoints &intersection_points, Index n_tags,
    Index n_intersection_points,
    const tf::buffer<tf::point<Int, 3>> &extra_points,
    const tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>>
        &splits,
    const tf::buffer<tf::cut::detail::region_triangulation_merge<Index>>
        &merges,
    tf::cut::detail::region_triangulation_workspace<Index, Int> &workspace,
    tf::buffer<std::array<tf::intersect::graph::vertex<Index>, 3>>
        &triangles,
    bool use_splits,
    tf::cut::detail::region_build build =
        tf::cut::detail::region_build::preserve,
    const InteriorPoints &interior_points = {}) -> bool {
  const bool assemble_only =
      build == tf::cut::detail::region_build::assemble;
  const bool resolve_crossings =
      build == tf::cut::detail::region_build::resolve;
  const Index tag = descriptor.tag;
  auto point = [&](const tf::intersect::graph::vertex<Index> &vertex) {
    return tf::cut::region_triangulation_point(
        n_intersection_points, extra_points, tag, vertex,
        intersection_points, get_mesh_point);
  };
  bool ok = true;
  auto holes = regions.holes();
  if (workspace.memo_tag != tag ||
      workspace.memo_object != descriptor.object) {
    apply_to_face(tag, descriptor.object, [&](const auto &face) {
      workspace.face_size = static_cast<Index>(face.size());
      const auto a = point(tf::intersect::graph::vertex<Index>{
          tf::intersect::graph::vertex_source::original,
          Index(face[0]),
          {0, tf::topo_type::vertex}});
      const auto b = point(tf::intersect::graph::vertex<Index>{
          tf::intersect::graph::vertex_source::original,
          Index(face[1]),
          {1, tf::topo_type::vertex}});
      const auto c = point(tf::intersect::graph::vertex<Index>{
          tf::intersect::graph::vertex_source::original,
          Index(face[2]),
          {2, tf::topo_type::vertex}});
      const auto axes = tf::exact::projection_axes(a, b, c);
      workspace.ax0 = int(axes.first);
      workspace.ax1 = int(axes.second);
      typename tf::exact::meta<Int>::T1 u[3];
      typename tf::exact::meta<Int>::T1 v[3];
      for (int coordinate = 0; coordinate < 3; ++coordinate) {
        u[coordinate] =
            typename tf::exact::meta<Int>::T1(b[coordinate]) -
            typename tf::exact::meta<Int>::T1(a[coordinate]);
        v[coordinate] =
            typename tf::exact::meta<Int>::T1(c[coordinate]) -
            typename tf::exact::meta<Int>::T1(a[coordinate]);
      }
      workspace.plane_n = {
          typename tf::exact::meta<Int>::T2(u[1]) *
                  typename tf::exact::meta<Int>::T2(v[2]) -
              typename tf::exact::meta<Int>::T2(u[2]) *
                  typename tf::exact::meta<Int>::T2(v[1]),
          typename tf::exact::meta<Int>::T2(u[2]) *
                  typename tf::exact::meta<Int>::T2(v[0]) -
              typename tf::exact::meta<Int>::T2(u[0]) *
                  typename tf::exact::meta<Int>::T2(v[2]),
          typename tf::exact::meta<Int>::T2(u[0]) *
                  typename tf::exact::meta<Int>::T2(v[1]) -
              typename tf::exact::meta<Int>::T2(u[1]) *
                  typename tf::exact::meta<Int>::T2(v[0])};
      workspace.plane_d =
          workspace.plane_n[0] *
              typename tf::exact::meta<Int>::T2(a[0]) +
          workspace.plane_n[1] *
              typename tf::exact::meta<Int>::T2(a[1]) +
          workspace.plane_n[2] *
              typename tf::exact::meta<Int>::T2(a[2]);
      workspace.plane_fallback =
          a[3 - workspace.ax0 - workspace.ax1];
    });
    workspace.memo_tag = tag;
    workspace.memo_object = descriptor.object;
  }

  const std::pair<int, int> axes{workspace.ax0, workspace.ax1};
  auto projected =
      [&](const tf::intersect::graph::vertex<Index> &vertex)
      -> tf::point<Int, 2> {
    const auto position = point(vertex);
    return {position[axes.first], position[axes.second]};
  };

  if (workspace.ihm.f().size())
    workspace.ihm.f().clear();
  workspace.ihm.kept_ids().clear();
  workspace.pts.clear();
  workspace.pos_set.clear();
  // Interior points are free vertices no constraint names, so they cannot
  // ride the positional fast path, which addresses the walk directly.
  const bool positional = !use_splits && hole_ids.size() == 0 &&
                          merges.size() == 0 && interior_points.size() == 0;
  auto append_at = [&](const tf::intersect::graph::vertex<Index> &vertex,
                       const tf::point<Int, 2> &position) -> Index {
    const auto id =
        static_cast<Index>(workspace.ihm.kept_ids().size());
    workspace.ihm.kept_ids().push_back(vertex);
    workspace.pts.push_back(position);
    workspace.pos_set.push_back(0);
    return id;
  };
  auto append =
      [&](const tf::intersect::graph::vertex<Index> &vertex) -> Index {
    return append_at(vertex, projected(vertex));
  };
  auto local_id =
      [&](const tf::intersect::graph::vertex<Index> &input) -> Index {
    auto vertex =
        tf::cut::resolve_region_vertex(n_tags, tag, input, merges);
    if (vertex.source != input.source || vertex.id != input.id)
      vertex.sub_id = {0, tf::topo_type::face};
    if (assemble_only)
      return append(vertex);
    auto found = workspace.ihm.f().find(vertex);
    if (found != workspace.ihm.f().end())
      return found->second;
    const auto id = append(vertex);
    workspace.ihm.f()[vertex] = id;
    return id;
  };
  workspace.cons.clear();
  workspace.cons_verts.clear();
  workspace.cons_slot.clear();
  workspace.cons_sub.clear();
  workspace.cons_parent.clear();
  workspace.cons_t.clear();
  workspace.split_seen.clear();
  typename tf::exact::meta<Int>::T2 area2(0);
  Index walk_base = 0;
  Index current_slot = 0;
  bool in_split = false;
  std::array<tf::intersect::graph::vertex<Index>, 2> current_parent{};
  std::array<typename tf::exact::meta<Int>::param_type, 2>
      current_parameters{};
  auto add_edge =
      [&](const tf::intersect::graph::vertex<Index> &a,
          const tf::intersect::graph::vertex<Index> &b) {
    const Index a_id = local_id(a);
    const Index b_id = local_id(b);
    if (a_id == b_id)
      return;
    const auto a_point = workspace.pts[std::size_t(a_id)];
    const auto b_point = workspace.pts[std::size_t(b_id)];
    if (a_point[0] == b_point[0] && a_point[1] == b_point[1])
      return;
    workspace.cons.push_back(a_id);
    workspace.cons.push_back(b_id);
    workspace.cons_verts.push_back({a, b});
    workspace.cons_slot.push_back(current_slot);
    workspace.cons_sub.push_back(char(in_split));
    workspace.cons_parent.push_back(
        in_split
            ? current_parent
            : std::array<tf::intersect::graph::vertex<Index>, 2>{a, b});
    workspace.cons_t.push_back(
        in_split ? current_parameters
                 : std::array<typename tf::exact::meta<Int>::param_type, 2>{
                       typename tf::exact::meta<Int>::param_type(0),
                       typename tf::exact::meta<Int>::param_type(0)});
  };
  auto add_walk = [&](const auto &walk, bool with_area) {
    const auto size = walk.size();
    for (std::size_t i = 0, j = size - 1; i < size; j = i++) {
      const auto &a = walk[j];
      const auto &b = walk[i];
      current_slot = walk_base + Index(j);
      if (with_area) {
        const auto p = projected(a);
        const auto q = projected(b);
        area2 +=
            typename tf::exact::meta<Int>::T2(
                typename tf::exact::meta<Int>::T1(p[0]) *
                    typename tf::exact::meta<Int>::T1(q[1]) -
                typename tf::exact::meta<Int>::T1(q[0]) *
                    typename tf::exact::meta<Int>::T1(p[1]));
      }
      if (!use_splits) {
        add_edge(a, b);
        continue;
      }
      const auto a_key =
          tf::cut::detail::region_vertex_key(n_tags, tag, a);
      const auto b_key =
          tf::cut::detail::region_vertex_key(n_tags, tag, b);
      const auto split_range = tf::cut::detail::find_region_edge_splits(
          tf::cut::detail::region_edge_key(a_key, b_key), splits);
      if (split_range.first == split_range.second) {
        add_edge(a, b);
        continue;
      }
      const bool forward = a_key < b_key;
      const auto a_point = point(a);
      const auto b_point = point(b);
      const auto &low = forward ? a_point : b_point;
      const auto &high = forward ? b_point : a_point;
      tf::topo_id<short> split_sub{0, tf::topo_type::face};
      if (auto face_edge = tf::cut::resolve_face_edge(
              a.sub_id, b.sub_id, std::size_t(workspace.face_size)))
        split_sub = {*face_edge, tf::topo_type::edge};
      auto previous = a;
      const typename tf::exact::meta<Int>::param_type maximum_parameter =
          typename tf::exact::meta<Int>::param_type(1)
          << tf::exact::meta<Int>::split_grid_bits;
      current_parent = {a, b};
      typename tf::exact::meta<Int>::param_type previous_parameter =
          forward ? typename tf::exact::meta<Int>::param_type(0)
                  : maximum_parameter;
      for (Index offset = 0;
           offset < split_range.second - split_range.first; ++offset) {
        const auto &split =
            splits[std::size_t(forward
                                   ? split_range.first + offset
                                   : split_range.second - 1 - offset)];
        auto split_vertex = split.vertex;
        split_vertex.sub_id = split_sub;
        Index id = Index(-1);
        const auto split_key = tf::cut::detail::region_vertex_key(
            n_tags, tag, split_vertex);
        if (assemble_only) {
          for (const auto &seen : workspace.split_seen)
            if (seen[0] == split_key[0] && seen[1] == split_key[1]) {
              id = seen[2];
              break;
            }
          if (id == Index(-1)) {
            id = local_id(split_vertex);
            workspace.split_seen.push_back(
                {split_key[0], split_key[1], id});
          } else {
            workspace.pts[std::size_t(id)] = projected(split_vertex);
            workspace.pos_set[std::size_t(id)] = 1;
          }
        } else {
          id = local_id(split_vertex);
        }
        if ((assemble_only && bool(split.named)) ||
            workspace.pos_set[std::size_t(id)]) {
          workspace.pts[std::size_t(id)] = projected(split_vertex);
          workspace.pos_set[std::size_t(id)] = 1;
        } else {
          workspace.pts[std::size_t(id)] =
              tf::cut::region_split_point(
                  low, high, split.param, axes.first, axes.second);
          workspace.pos_set[std::size_t(id)] = 1;
        }
        current_parameters = {previous_parameter, split.param};
        in_split = true;
        add_edge(previous, split_vertex);
        in_split = false;
        previous_parameter = split.param;
        previous = split_vertex;
      }
      current_parameters = {
          previous_parameter,
          forward ? maximum_parameter
                  : typename tf::exact::meta<Int>::param_type(0)};
      in_split = true;
      add_edge(previous, b);
      in_split = false;
    }
    walk_base += Index(size);
  };

  if (positional) {
    const auto size = loop.size();
    for (std::size_t i = 0; i < size; ++i)
      workspace.pts.push_back(projected(loop[i]));
    for (std::size_t i = 0, j = size - 1; i < size; j = i++) {
      const auto p = workspace.pts[j];
      const auto q = workspace.pts[i];
      area2 += typename tf::exact::meta<Int>::T2(
          typename tf::exact::meta<Int>::T1(p[0]) *
                  typename tf::exact::meta<Int>::T1(q[1]) -
              typename tf::exact::meta<Int>::T1(q[0]) *
                  typename tf::exact::meta<Int>::T1(p[1]));
      if (p[0] == q[0] && p[1] == q[1])
        continue;
      workspace.cons.push_back(Index(j));
      workspace.cons.push_back(Index(i));
      workspace.cons_verts.push_back({loop[j], loop[i]});
      workspace.cons_slot.push_back(Index(j));
      workspace.cons_sub.push_back(char(0));
      workspace.cons_parent.push_back({loop[j], loop[i]});
      workspace.cons_t.push_back({Index(0), Index(0)});
    }
  } else {
    add_walk(loop, true);
    for (auto hole : hole_ids)
      add_walk(holes[hole], false);
  }

  const bool triangle_region = !use_splits && hole_ids.size() == 0 &&
                               loop.size() == 3 &&
                               interior_points.size() == 0;
  std::array<Index, 3> triangle_inputs{};
  if (triangle_region)
    for (std::size_t corner = 0; corner < 3; ++corner)
      triangle_inputs[corner] =
          positional ? Index(corner) : local_id(loop[corner]);

  // Points the region carries that no constraint refers to — the Steiner
  // points refinement chose. They enter as ordinary vertices, which is the
  // whole of the difference between a refined region and a plain one.
  for (const auto &interior : interior_points)
    append_at(interior.vertex, {interior.position[axes.first],
                                interior.position[axes.second]});

  workspace.flip = typename tf::exact::meta<Int>::T2(0) < area2;
  if (assemble_only)
    return ok;

  const auto &kept_ids = workspace.ihm.kept_ids();
  const std::size_t n_input =
      positional ? loop.size() : kept_ids.size();
  auto vertex_at =
      [&](std::size_t index)
      -> const tf::intersect::graph::vertex<Index> & {
    return positional ? loop[index] : kept_ids[index];
  };
  auto edges = tf::make_edges(tf::make_blocked_range<2>(
      tf::make_range(workspace.cons.data(), workspace.cons.size())));
  workspace.tri.clear();
  const bool built = workspace.tri.build(
      tf::make_points(workspace.pts), edges,
      tf::make_constant_range(true, edges.size()), resolve_crossings);

  const auto &input_to_output = workspace.tri.index_map().f();
  const auto &output_to_input = workspace.tri.index_map().kept_ids();
  const Index n_final = Index(workspace.tri.points().size());
  Index n_created = 0;
  if (resolve_crossings)
    for (Index output = 0; output < n_final; ++output)
      if (output_to_input[std::size_t(output)] == Index(n_input))
        ++n_created;
  // Counts alone cannot detect a moved identity: a resolving build that
  // welds one vertex and creates one point leaves the count unchanged.
  const bool remapped =
      std::size_t(n_final) != n_input || n_created != Index(0);
  workspace.rep.clear();
  if (remapped) {
    workspace.rep.allocate(std::size_t(n_final));
    std::fill(workspace.rep.begin(), workspace.rep.end(), Index(-1));
    for (Index input = 0; input < Index(n_input); ++input) {
      const Index output = input_to_output[std::size_t(input)];
      if (output < 0 || n_final <= output)
        return false;
      auto &representative = workspace.rep[std::size_t(output)];
      if (representative == Index(-1) ||
          tf::cut::detail::region_vertex_key(
              n_tags, tag, vertex_at(std::size_t(input))) <
              tf::cut::detail::region_vertex_key(
                  n_tags, tag,
                  vertex_at(std::size_t(representative))))
        representative = input;
    }
    for (Index input = 0; input < Index(n_input); ++input) {
      const Index representative =
          workspace.rep[std::size_t(
              input_to_output[std::size_t(input)])];
      if (representative != Index(-1) && representative != input)
        workspace.merges.push_back({
            tf::cut::detail::region_vertex_key(
                n_tags, tag, vertex_at(std::size_t(input))),
            tf::cut::detail::region_vertex_key(
                n_tags, tag,
                vertex_at(std::size_t(representative))),
            vertex_at(std::size_t(representative))});
    }
  }

  if (triangle_region) {
    std::array<Index, 3> final{};
    for (std::size_t corner = 0; corner < 3; ++corner) {
      final[corner] =
          input_to_output[std::size_t(triangle_inputs[corner])];
      if (final[corner] < 0 || n_final <= final[corner])
        return false;
    }
    if (final[0] == final[1] || final[1] == final[2] ||
        final[0] == final[2])
      return true;
    std::array<tf::intersect::graph::vertex<Index>, 3> triangle;
    for (std::size_t corner = 0; corner < 3; ++corner) {
      const Index representative =
          remapped ? workspace.rep[std::size_t(final[corner])]
                   : output_to_input[std::size_t(final[corner])];
      if (representative == Index(-1))
        return false;
      triangle[corner] = vertex_at(std::size_t(representative));
    }
    triangles.push_back(triangle);
    return true;
  }

  if (!built)
    return false;
  // Only a completed resolving build speaks: welds above are facts of the
  // input points, but an identification is derived from the build's own
  // crossing records, and a refused build's records name a state that was
  // abandoned.
  if (n_created != Index(0))
    tf::cut::identify_resolved_crossings<Index, Int>(
        workspace, n_tags, tag, Index(n_input), vertex_at, point);
  auto faces = workspace.tri.make_faces();
  auto labels = workspace.tri.region_labels();
  const bool flip = typename tf::exact::meta<Int>::T2(0) < area2;
  workspace.flip = flip;
  std::size_t face_index = 0;
  for (auto face : faces) {
    if ((labels[face_index++] & 1) == 0)
      continue;
    const Index a =
        remapped ? workspace.rep[std::size_t(face[0])]
                 : output_to_input[std::size_t(face[0])];
    const Index b =
        remapped ? workspace.rep[std::size_t(face[1])]
                 : output_to_input[std::size_t(face[1])];
    const Index c =
        remapped ? workspace.rep[std::size_t(face[2])]
                 : output_to_input[std::size_t(face[2])];
    if (a == Index(-1) || b == Index(-1) || c == Index(-1))
      return false;
    // An identified crossing can pull two corners onto one vertex; the
    // triangle it leaves has no area and no reader.
    if (a == b || b == c || a == c)
      continue;
    if (flip)
      triangles.push_back({vertex_at(std::size_t(a)),
                           vertex_at(std::size_t(c)),
                           vertex_at(std::size_t(b))});
    else
      triangles.push_back({vertex_at(std::size_t(a)),
                           vertex_at(std::size_t(b)),
                           vertex_at(std::size_t(c))});
  }
  return true;
}

} // namespace tf::cut
