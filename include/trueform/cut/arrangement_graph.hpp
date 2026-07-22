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
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/frame_of.hpp"
#include "../core/none.hpp"
#include "../core/point.hpp"
#include "../core/range.hpp"
#include "../core/transformed.hpp"
#include "../core/views/sequence_range.hpp"
#include "../exact/resolve_int_type.hpp"
#include "../exact/vertex_converter.hpp"
#include "../intersect/exact/make_kernel.hpp"
#include "../intersect/graph/intersection_graph.hpp"
#include "../intersect/intersect_config.hpp"
#include "./arrangement_config.hpp"
#include "../intersect/polygon_intersections.hpp"
#include "../topology/triangulation_type.hpp"
#include "./dispatch/arrangement_range_policy.hpp"
#include "./face_regions.hpp"
#include "./make_coplanar_loop_pairs.hpp"
#include "./region_triangulator.hpp"

#include "tbb/parallel_sort.h"
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup cut
/// @brief The arrangement of a set of forms — everything below
///        classification: intersections, the intersection graph, the
///        face-region structure, the coplanar stacks (pairs + dead
///        mask, detected once here), and the region triangulation with
///        its unified created-points table. The graph exposes the
///        arrangement at TRIANGLE grain: every loop of `loops()` is one
///        triangle of the finalized stream, so nothing downstream ever
///        triangulates. Loop connectivity is classification-tier —
///        @ref tf::cut::component_labels owns it.
///
/// Self-contained: stores the user's `forms` and (if the dispatch
/// layer detected missing tags) the owned computed structures;
/// `forms()` rebuilds the tagged view on demand.
///
/// @ref tf::csg_graph is an arrangement_graph plus classification
/// (descriptor, inclusions, volumes, sheet anchoring).
///
/// Template parameters mirror @ref tf::csg_graph: `Forms`/`Structs`
/// are deduced from the constructor, `Int` is the exact-integer
/// override resolved from the form's coordinate type.
template <typename Policy, typename Int = tf::none_t>
class arrangement_graph {
public:
  using policy_type = Policy;
  using index_type = typename Policy::index_type;
  using input_real_type = typename Policy::input_real_type;
  using resolved_int_type = tf::exact::resolve_int_type<Int, input_real_type>;
  using pipeline_real_type =
      std::conditional_t<std::is_integral_v<input_real_type>, input_real_type,
                         double>;

  arrangement_graph(Policy policy, tf::arrangement_config config = {})
      : _policy(std::move(policy)) {
    _build(config);
  }

  /// @brief The tagged forms view (policies that store a homogeneous
  ///        range; the pair policy exposes `apply_to_form` instead).
  auto forms() const { return _policy.forms(); }

  /// @brief Two-phase form access: `apply_to_form()(tag, f)`.
  auto apply_to_form() const { return _policy.make_apply_to_form(); }

  auto n_tags() const -> index_type { return _policy.n_tags(); }

  auto converter() const -> const
      tf::exact::vertex_converter<resolved_int_type, pipeline_real_type, 3> & {
    return _intersections.converter();
  }
  auto intersections() const
      -> const tf::polygon_intersections<index_type, pipeline_real_type,
                                         resolved_int_type> & {
    return _intersections;
  }
  auto intersection_graph() const
      -> const tf::intersection_graph<index_type, resolved_int_type> & {
    return _ig;
  }

  /// @brief The raw region structure the triangulation was built from.
  auto face_regions() const
      -> const tf::face_regions<index_type, resolved_int_type> & {
    return _fr;
  }

  /// @brief The region triangulation — the owner of the exposed
  ///        triangle-grain stream and of provenance the stream alone
  ///        cannot carry (`merges()`, `original_edge_splits()`,
  ///        `promoted_descriptors()`).
  auto triangulations() const
      -> const tf::cut::region_triangulator<index_type, resolved_int_type> & {
    return _rt;
  }

  /// @brief The exposed loops: one triangle each, contiguous per tag,
  ///        dead coplanar members re-emitted under their own
  ///        descriptors (winding already applied) and promoted faces
  ///        after each tag's structure loops.
  auto loops() const { return _rt.loops(); }
  /// @brief One descriptor per exposed loop, from its owning region.
  auto descriptors() const { return _rt.descriptors(); }
  /// @brief Per-tag `[begin, end)` offsets into `loops()`.
  auto tag_offsets() const { return _rt.tag_offsets(); }
  /// @brief Per exposed face `[begin, end)` offsets into `loops()`.
  auto face_offsets() const { return _rt.face_offsets(); }
  /// @brief Cut-and-consumed faces of form `tag` — cut, never
  ///        untouched originals.
  auto deleted(index_type tag) const { return _rt.deleted(tag); }
  auto deleted_offsets() const { return _rt.deleted_offsets(); }

  /// @brief Unified created-points table (int lattice): the
  ///        intersection graph's points followed by any points the
  ///        triangulation materialized (recovery splits, refinement).
  ///        Created vertex ids index this buffer directly.
  auto created_points() const
      -> const tf::buffer<tf::point<resolved_int_type, 3>> & {
    return _created_points;
  }

  /// True when the build carried @ref tf::intersect_mode::within —
  /// operands may self-overlap, so parity bits cannot be trusted for
  /// domain extraction.
  auto with_self() const -> bool { return _with_self; }

  /// @brief The configuration this graph was built with.
  auto config() const -> const tf::arrangement_config & { return _config; }

  /// @brief Region-grain stack triples {survivor, dead, reversed},
  ///        sorted — the classification layer's currency; triangles
  ///        inherit through `triangulations().exposed_ranges()`
  ///        (region-major scatters).
  auto coplanar_pairs() const { return tf::make_range(_coplanar_pairs); }

  /// @brief Per REGION dead flag: `1` = region of a coplanar
  ///        duplicate, folded onto its stack's survivor. Triangles
  ///        inherit through `triangulations().exposed_ranges()`.
  auto dead_loops() const { return tf::make_range(_dead_loops); }

private:
  auto _build(tf::arrangement_config a_config) -> void {
    _config = a_config;
    auto config = a_config.intersect;
    // One form has no pairs: the graph IS the self arrangement, so the
    // build runs within (and thereby self-contour resolution).
    if (_policy.n_tags() == index_type(1))
      config.mode = config.mode | tf::intersect_mode::within;
    _with_self = bool(config.mode & tf::intersect_mode::self_intersections);
    _policy.build_intersections(_intersections, config);
    auto &conv = _intersections.converter();

    auto apply_form = _policy.make_apply_to_form();
    auto apply_to_face = [apply_form](int tag, index_type object,
                                      const auto &f) {
      apply_form(index_type(tag),
                 [&](const auto &form) { f(form.faces()[object]); });
    };
    auto get_mesh_point =
        [apply_form, &conv](int tag,
                            index_type id) -> tf::point<resolved_int_type, 3> {
      tf::point<resolved_int_type, 3> out;
      apply_form(index_type(tag), [&](const auto &form) {
        out = conv.convert(
            tf::transformed(form.points()[id], tf::frame_of(form)));
      });
      return out;
    };

    _ig.build(_intersections, apply_to_face, get_mesh_point, config.mode,
              tf::exact::make_kernel(conv, config.tolerance));
    _fr.build(_ig, apply_to_face, get_mesh_point);

    // Coplanar stacks: the one detection, at REGION grain — the
    // triangulation folds dead members onto their survivors from it,
    // and the classification layer reads the triples as-is.
    auto &dead_regions = _dead_loops;
    auto &region_pairs = _coplanar_pairs;
    {
      auto pairs = tf::cut::make_coplanar_loop_pairs_all(_fr);
      region_pairs.allocate(std::size_t(pairs.size()));
      dead_regions.allocate(std::size_t(_fr.loops().size()));
      tf::parallel_fill(dead_regions, char(0));
      for (std::size_t i = 0; i < std::size_t(pairs.size()); ++i) {
        region_pairs[i] = {pairs[i].loop_a, pairs[i].loop_b,
                           index_type(pairs[i].opposing)};
        dead_regions[std::size_t(pairs[i].loop_b)] = char(1);
      }
      tbb::parallel_sort(region_pairs.begin(), region_pairs.end());
    }

    _rt.build_applied(_fr, _ig, apply_form, apply_to_face, get_mesh_point,
                      tf::make_range(dead_regions),
                      tf::make_range(region_pairs));
    if (a_config.triangulation == tf::triangulation_type::refined_cdt) {
      tf::cdt_refine_config rc;
      rc.min_quality = 0.3f;
      _rt.refine_applied(_fr, _ig, _policy.n_tags(), apply_form,
                         apply_to_face, get_mesh_point, rc,
                         tf::make_range(dead_regions),
                         tf::make_range(region_pairs));
    }

    auto ig_pts = _ig.points();
    _created_points.allocate(std::size_t(ig_pts.size()) +
                             _rt.extra_points.size());
    tf::parallel_for_each(
        tf::make_sequence_range(std::size_t(ig_pts.size())),
        [&](std::size_t i) { _created_points[i] = ig_pts[index_type(i)]; });
    tf::parallel_copy(
        tf::make_range(_rt.extra_points),
        tf::make_range(_created_points.begin() + std::size_t(ig_pts.size()),
                       _created_points.end()));
  }

  Policy _policy;
  bool _with_self = false;
  tf::arrangement_config _config;
  tf::polygon_intersections<index_type, pipeline_real_type, resolved_int_type>
      _intersections;
  tf::intersection_graph<index_type, resolved_int_type> _ig;
  tf::face_regions<index_type, resolved_int_type> _fr;
  tf::cut::region_triangulator<index_type, resolved_int_type> _rt;
  tf::buffer<std::array<index_type, 3>> _coplanar_pairs;
  tf::buffer<char> _dead_loops;
  tf::buffer<tf::point<resolved_int_type, 3>> _created_points;
};

} // namespace tf
