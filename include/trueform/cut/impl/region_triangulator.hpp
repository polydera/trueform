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
#include "../../core/range.hpp"
#include "../../core/views/slice.hpp"
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/intersection_graph.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../topology/cdt_refine_config.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "../face_regions.hpp"
#include "./apply_region_stack_aliases.hpp"
#include "./conform_region_welds.hpp"
#include "./consolidate_region_merges.hpp"
#include "./discover_region_refinement.hpp"
#include "./for_each_region_edge_split.hpp"
#include "./make_original_edge_splits.hpp"
#include "./make_region_refinement_fingerprints.hpp"
#include "./make_region_triangulation_exposure.hpp"
#include "./materialize_region_split_proposals.hpp"
#include "./materialize_region_steiners.hpp"
#include "./rebuild_region_steiners.hpp"
#include "./persist_promoted_face_steiners.hpp"
#include "./promote_split_faces.hpp"
#include "./propagate_conforming_rings.hpp"
#include "./recover_region_triangulations.hpp"
#include "./settle_region_triangulations.hpp"
#include "./resolve_region_vertex.hpp"
#include "./triangulate_region_blocks.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::cut {

template <typename Index, typename Int = tf::exact::int32>
class region_triangulator {
public:
  struct original_edge_split {
    Index tag;
    Index u;
    Index v;
    tf::intersect::graph::vertex<Index> vertex;
  };

  tf::buffer<std::array<Index, 2>> ranges;
  tf::buffer<char> rev;
  tf::buffer<
      std::array<tf::intersect::graph::vertex<Index>, 3>>
      tris;
  tf::buffer<Index> failed;
  Index n_failed = 0;
  Index n_refine_refused = 0;
  Index n_refine_recovered = 0;
  tf::buffer<tf::point<Int, 3>> extra_points;
  Index n_ig_created = 0;

  auto original_edge_splits() const {
    return tf::make_range(_original_splits);
  }

  auto promoted_descriptors() const {
    return tf::make_range(_promoted);
  }

  auto n_tags() const -> Index { return _n_tags; }

  auto promoted_index(Index tag, Index object) const -> Index {
    auto it = std::lower_bound(
        _promoted.begin(), _promoted.end(),
        tf::intersect::graph::face_descriptor<Index>{tag, object},
        [](const tf::intersect::graph::face_descriptor<Index> &a,
           const tf::intersect::graph::face_descriptor<Index> &b) {
          return a.tag != b.tag ? a.tag < b.tag : a.object < b.object;
        });
    if (it == _promoted.end() || it->tag != tag || it->object != object)
      return Index(-1);
    return Index(it - _promoted.begin());
  }

  auto promoted_range(Index promoted_index) const
      -> std::array<Index, 2> {
    const std::size_t n_regions =
        _exposed_ranges.size() - _promoted.size();
    return _exposed_ranges[n_regions +
                           std::size_t(promoted_index)];
  }

  auto merges() const { return tf::make_range(_merge); }

  auto loops() const {
    const auto &source =
        _exposed_owned ? _exposed_triangles : tris;
    return tf::make_range(source);
  }

  auto descriptors() const {
    return tf::make_range(_exposed_descriptors);
  }

  auto tag_offsets() const {
    return tf::make_range(_tag_offsets);
  }

  auto face_offsets() const {
    return tf::make_range(_face_offsets);
  }

  auto exposed_ranges() const {
    return tf::make_range(_exposed_ranges);
  }

  auto deleted(Index tag) const {
    return tf::slice(
        tf::make_range(_deleted),
        std::size_t(_deleted_offsets[std::size_t(tag)]),
        std::size_t(_deleted_offsets[std::size_t(tag) + 1]));
  }

  auto deleted_offsets() const {
    return tf::make_range(_deleted_offsets);
  }

  auto loop_tris(Index loop) const {
    const auto range = ranges[std::size_t(loop)];
    return tf::make_range(tris.begin() + range[0],
                          tris.begin() + range[1]);
  }

  auto clear() -> void {
    ranges.clear();
    rev.clear();
    tris.clear();
    failed.clear();
    n_failed = 0;
    n_refine_refused = 0;
    n_refine_recovered = 0;
    extra_points.clear();
    n_ig_created = 0;
    _n_tags = 0;
    _splits.clear();
    _original_splits.clear();
    _promoted.clear();
    _merge.clear();
    _steiners.clear();
    _region_fingerprints.clear();
    _refined_config = tf::cdt_refine_config{};
    _refined_config.min_quality = -1.f;
    _promoted_steiners.clear();
    _promoted_interior_of.clear();
    _promoted_interior_base = 0;
    _aliased = false;
    _exposed_owned = false;
    _exposed_triangles.clear();
    _exposed_descriptors.clear();
    _exposed_ranges.clear();
    _face_offsets.clear();
    _tag_offsets.clear();
    _deleted.clear();
    _deleted_offsets.clear();
  }

  template <typename Forms, typename ApplyToFace,
            typename GetMeshPoint>
  auto build(
      const tf::face_regions<Index, Int> &regions,
      const tf::intersection_graph<Index, Int> &intersection_graph,
      const Forms &forms, const ApplyToFace &apply_to_face,
      const GetMeshPoint &get_mesh_point) -> void {
    const tf::buffer<char> no_dead;
    const tf::buffer<std::array<Index, 3>> no_pairs;
    build(regions, intersection_graph, forms, apply_to_face,
          get_mesh_point, tf::make_range(no_dead),
          tf::make_range(no_pairs));
  }

  template <typename Forms, typename ApplyToFace,
            typename GetMeshPoint, typename DeadLoops,
            typename CoplanarPairs>
  auto build(
      const tf::face_regions<Index, Int> &regions,
      const tf::intersection_graph<Index, Int> &intersection_graph,
      const Forms &forms, const ApplyToFace &apply_to_face,
      const GetMeshPoint &get_mesh_point,
      const DeadLoops &dead_loops,
      const CoplanarPairs &coplanar_pairs) -> void {
    auto apply_to_form = [&forms](Index tag, auto &&apply) {
      apply(forms[tag]);
    };
    build_applied(regions, intersection_graph, apply_to_form,
                  apply_to_face, get_mesh_point, dead_loops,
                  coplanar_pairs);
  }

  template <typename ApplyToForm, typename ApplyToFace,
            typename GetMeshPoint, typename DeadLoops,
            typename CoplanarPairs>
  auto build_applied(
      const tf::face_regions<Index, Int> &regions,
      const tf::intersection_graph<Index, Int> &intersection_graph,
      const ApplyToForm &apply_to_form,
      const ApplyToFace &apply_to_face,
      const GetMeshPoint &get_mesh_point,
      const DeadLoops &dead_loops,
      const CoplanarPairs &coplanar_pairs) -> void {
    clear();
    const auto intersection_points = intersection_graph.points();
    n_ig_created = static_cast<Index>(intersection_points.size());
    _n_tags =
        static_cast<Index>(regions.tag_offsets().size()) - 1;

    tf::buffer<
        tf::cut::detail::region_triangulation_merge<Index>>
        pending_merges;
    tf::cut::triangulate_region_blocks(
        regions, apply_to_face, get_mesh_point,
        intersection_points, _n_tags, n_ig_created,
        extra_points, _splits, _merge, dead_loops, ranges,
        tris, failed, pending_merges);
    tf::cut::consolidate_region_merges<Index>(
        _merge, pending_merges);

    if (failed.size() || _merge.size())
      tf::cut::recover_region_triangulations(
          regions, intersection_graph, apply_to_face,
          get_mesh_point, dead_loops, _n_tags,
          n_ig_created, extra_points, _splits, _merge,
          ranges, tris, failed);

    n_failed = static_cast<Index>(failed.size());
    tf::cut::make_original_edge_splits(
        _splits, _n_tags, _original_splits);

    tf::buffer<
        tf::cut::detail::region_triangulation_merge<Index>>
        promotion_merges;
    tf::cut::promote_split_faces(
        regions, _n_tags, apply_to_form, apply_to_face,
        get_mesh_point, intersection_points, n_ig_created,
        _original_splits, _promoted_steiners, _splits,
        _merge, extra_points, ranges, tris, failed, _promoted,
        _promoted_interior_of, _promoted_interior_base,
        promotion_merges);

    n_failed = tf::cut::settle_region_triangulations(
        regions, intersection_graph, apply_to_face,
        get_mesh_point, dead_loops, _n_tags, n_ig_created,
        extra_points, _splits, _merge, promotion_merges,
        _promoted, ranges, tris, failed);

    tf::cut::apply_region_stack_aliases(
        dead_loops, coplanar_pairs, ranges, rev, _aliased);
    finalize(regions);
  }

  template <typename Forms, typename ApplyToFace,
            typename GetMeshPoint>
  auto refine(
      const tf::face_regions<Index, Int> &regions,
      const tf::intersection_graph<Index, Int> &intersection_graph,
      const Forms &forms, const ApplyToFace &apply_to_face,
      const GetMeshPoint &get_mesh_point,
      const tf::cdt_refine_config &config) -> void {
    const tf::buffer<char> no_dead;
    const tf::buffer<std::array<Index, 3>> no_pairs;
    auto apply_to_form = [&forms](Index tag, auto &&apply) {
      apply(forms[tag]);
    };
    refine_applied(
        regions, intersection_graph, apply_to_form,
        apply_to_face, get_mesh_point, config,
        tf::make_range(no_dead), tf::make_range(no_pairs));
  }

  template <typename ApplyToForm, typename ApplyToFace,
            typename GetMeshPoint, typename DeadLoops,
            typename CoplanarPairs>
  auto refine_applied(
      const tf::face_regions<Index, Int> &regions,
      const tf::intersection_graph<Index, Int> &intersection_graph,
      const ApplyToForm &apply_to_form,
      const ApplyToFace &apply_to_face,
      const GetMeshPoint &get_mesh_point,
      const tf::cdt_refine_config &config,
      const DeadLoops &dead_loops,
      const CoplanarPairs &coplanar_pairs) -> void {
    const auto loops = regions.loops();
    const auto intersection_points = intersection_graph.points();
    const std::size_t n_splits_before = _splits.size();

    auto discovery = tf::cut::discover_region_refinement(
        regions, _n_tags, apply_to_form, apply_to_face,
        get_mesh_point, intersection_points, config, dead_loops,
        n_ig_created, _region_fingerprints, _refined_config,
        _steiners, _merge, extra_points, _splits);
    n_refine_refused = discovery.refused;
    n_refine_recovered = discovery.recovered;
    auto proposals = std::move(discovery.proposals);
    auto steiner_points = std::move(discovery.steiner_points);
    auto steiner_counts = std::move(discovery.steiner_counts);

    tf::cut::materialize_region_split_proposals(
        _n_tags, n_ig_created, proposals, intersection_points,
        get_mesh_point, extra_points, _splits);

    if (_splits.size() == n_splits_before &&
        steiner_points.size() == 0)
      return;

    tf::cut::make_original_edge_splits(
        _splits, _n_tags, _original_splits);
    tf::cut::propagate_conforming_rings(
        regions, _n_tags, n_ig_created, apply_to_form,
        get_mesh_point, intersection_points, extra_points,
        _splits, _original_splits);

    auto materialized_steiners =
        tf::cut::materialize_region_steiners(
            std::size_t(loops.size()), steiner_counts,
            steiner_points, extra_points);
    tf::cut::rebuild_region_steiners(
        std::size_t(loops.size()), steiner_counts,
        n_ig_created + materialized_steiners.base, _steiners);

    // Phase two is the stock build, unchanged. The only thing refinement
    // did was choose the splits and the interior points it now hands over;
    // one mechanism triangulates, and one wave makes it succeed.
    tf::buffer<
        tf::cut::detail::region_triangulation_merge<Index>>
        pending_merges;
    tf::cut::triangulate_region_blocks(
        regions, apply_to_face, get_mesh_point,
        intersection_points, _n_tags, n_ig_created,
        extra_points, _splits, _merge, dead_loops, ranges,
        tris, failed, pending_merges, _steiners);
    tf::cut::consolidate_region_merges<Index>(
        _merge, pending_merges);

    if (failed.size() || _merge.size())
      tf::cut::recover_region_triangulations(
          regions, intersection_graph, apply_to_face,
          get_mesh_point, dead_loops, _n_tags, n_ig_created,
          extra_points, _splits, _merge, ranges, tris, failed,
          _steiners);

    tf::cut::make_original_edge_splits(
        _splits, _n_tags, _original_splits);
    _promoted.clear();

    tf::cdt_refine_config promoted_config = config;
    promoted_config.split_encroached = false;
    tf::cut::promote_split_faces(
        regions, _n_tags, apply_to_form, apply_to_face,
        get_mesh_point, intersection_points, n_ig_created,
        _original_splits, _promoted_steiners, _splits,
        _merge, promoted_config, extra_points, ranges, tris,
        failed, _promoted, _promoted_interior_of,
        _promoted_interior_base, pending_merges);

    n_failed = tf::cut::settle_region_triangulations(
        regions, intersection_graph, apply_to_face,
        get_mesh_point, dead_loops, _n_tags, n_ig_created,
        extra_points, _splits, _merge, pending_merges,
        _promoted, ranges, tris, failed, _steiners);

    tf::cut::persist_promoted_face_steiners(
        _n_tags, n_ig_created, _promoted_interior_base,
        _promoted, _promoted_interior_of, _merge,
        _promoted_steiners);
    tf::cut::apply_region_stack_aliases(
        dead_loops, coplanar_pairs, ranges, rev, _aliased);

    tf::cut::make_region_refinement_fingerprints(
        regions, _n_tags, _splits, _steiners,
        _region_fingerprints);
    _refined_config = config;
    finalize(regions);
  }

private:
  auto finalize(const tf::face_regions<Index, Int> &regions)
      -> void {
    tf::cut::make_region_triangulation_exposure(
        regions, _n_tags, ranges, rev, tris, _promoted,
        _aliased, _exposed_owned, _exposed_triangles,
        _exposed_descriptors, _exposed_ranges, _face_offsets,
        _tag_offsets, _deleted, _deleted_offsets);
  }

public:
  auto resolve_key(
      Index tag,
      const tf::intersect::graph::vertex<Index> &vertex) const
      -> std::array<Index, 2> {
    return tf::cut::resolve_region_vertex_key(
        _n_tags, tag, vertex, _merge);
  }

  auto resolve(
      Index tag,
      const tf::intersect::graph::vertex<Index> &vertex) const
      -> tf::intersect::graph::vertex<Index> {
    return tf::cut::resolve_region_vertex(
        _n_tags, tag, vertex, _merge);
  }

  template <typename Emit>
  auto for_each_edge_split(
      Index tag,
      const tf::intersect::graph::vertex<Index> &first,
      const tf::intersect::graph::vertex<Index> &second,
      const Emit &emit) const -> void {
    tf::cut::for_each_region_edge_split(
        _n_tags, tag, first, second, _splits, emit);
  }

private:
  Index _n_tags = 0;
  tf::buffer<
      tf::cut::detail::region_triangulation_split<Index, Int>>
      _splits;
  tf::buffer<original_edge_split> _original_splits;
  tf::buffer<tf::intersect::graph::face_descriptor<Index>>
      _promoted;
  tf::buffer<
      tf::cut::detail::region_triangulation_merge<Index>>
      _merge;
  tf::offset_block_buffer<Index, Index> _steiners;
  tf::buffer<std::array<Index, 2>> _region_fingerprints;
  tf::cdt_refine_config _refined_config = [] {
    tf::cdt_refine_config config;
    config.min_quality = -1.f;
    return config;
  }();
  tf::buffer<std::array<Index, 3>> _promoted_steiners;
  tf::buffer<Index> _promoted_interior_of;
  Index _promoted_interior_base = 0;
  bool _aliased = false;
  bool _exposed_owned = false;
  tf::buffer<
      std::array<tf::intersect::graph::vertex<Index>, 3>>
      _exposed_triangles;
  tf::buffer<tf::intersect::graph::face_descriptor<Index>>
      _exposed_descriptors;
  tf::buffer<std::array<Index, 2>> _exposed_ranges;
  tf::buffer<Index> _face_offsets;
  tf::buffer<Index> _tag_offsets;
  tf::buffer<Index> _deleted;
  tf::buffer<Index> _deleted_offsets;
};

} // namespace tf::cut
