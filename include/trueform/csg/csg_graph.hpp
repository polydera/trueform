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
#include "./graph/loop_triangulations.hpp"
#include "../topology/triangulation_type.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/frame_of.hpp"
#include "../core/none.hpp"
#include "../core/point.hpp"
#include "../core/small_vector.hpp"
#include "../core/transformed.hpp"
#include "../core/views/mapped_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "../core/views/zip.hpp"
#include "../cut/arrangement_graph.hpp"
#include "../cut/arrangements/anchor_sheet_sides.hpp"
#include "../cut/arrangements/arrangement_descriptor.hpp"
#include "../cut/arrangements/compute_domain_inclusions.hpp"
#include "../cut/arrangements/make_arrangement_descriptor.hpp"
#include "../cut/arrangements/propagate_inclusion_bits.hpp"
#include "../cut/face_cuts.hpp"
#include "../exact/resolve_int_type.hpp"
#include "../exact/vertex_converter.hpp"
#include "../intersect/exact/make_kernel.hpp"
#include "../intersect/graph/intersection_graph.hpp"
#include "../intersect/intersect_config.hpp"
#include "../intersect/polygon_intersections.hpp"
#include "./graph/compute_arrangement_domain_volumes.hpp"
#include "./graph/seed_inclusion_bits.hpp"
#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup csg
/// @brief Implicit N-form CSG graph — owns the arrangement state and
///        any auto-tagged structures for a set of forms.
///
/// Built once per form set, reused across many @ref tf::make_csg_mesh
/// calls (one per boolean expression).
///
/// Self-contained: the graph stores the user's `forms` and (if the
/// dispatch layer detected missing tags) the owned
/// `small_vector<Structs, 10>` of computed structures, exactly as
/// @ref tf::cut::dispatch::arrangement does. `forms()` rebuilds the
/// tagged view on demand by mapping over `(forms, structs)`.
///
/// The graph also owns the per-loop triangulation store (see
/// @ref tf::csg::graph::loop_triangulations) and the unified
/// created-points table the extractions read; `tri` selects the store
/// flavour (see @ref tf::triangulation_type).
///
/// Template parameters:
/// - `Forms`, `Structs` — deduced from the constructor.
/// - `Int` — exact-integer override (defaulted to `tf::none_t`,
///   resolved via @ref tf::exact::resolve_int_type from the form's
///   coordinate type).
///
/// `Index`, `InputReal`, `PipelineReal`, `ResolvedInt` are
/// **derived** from `Forms` + `Int` and exposed as nested aliases.
///
/// Output coordinate type is not on the graph — materialisation
/// happens in @ref tf::make_csg_mesh, which takes its own
/// `OutputCoordinateType` parameter.
///
/// @pre A one-form graph's form is a volume, not a declared sheet — a
///      lone sheet is a cutter with nothing to cut.
template <typename Forms, typename Structs, typename Int = tf::none_t>
class csg_graph {
public:
  using forms_type = Forms;
  using structs_type = Structs;

  using index_type =
      std::decay_t<decltype(std::declval<Forms>()[0].faces()[0][0])>;
  using input_real_type =
      tf::coordinate_type<decltype(std::declval<Forms>()[0])>;
  using resolved_int_type = tf::exact::resolve_int_type<Int, input_real_type>;
  using pipeline_real_type =
      std::conditional_t<std::is_integral_v<input_real_type>, input_real_type,
                         double>;

  csg_graph(Forms forms, tf::small_vector<Structs, 10> structs,
            tf::intersect_config config =
                {tf::intersect_mode::primitives |
                 tf::intersect_mode::resolve_crossing_contours},
            tf::buffer<char> is_sheet = {},
            tf::triangulation_type tri = tf::triangulation_type::cdt)
      : _forms(std::move(forms)), _structs(std::move(structs)),
        _is_sheet(std::move(is_sheet)) {
    auto tagged = this->forms();
    // One form has no pairs: the graph IS the self arrangement, so the
    // build runs within (and thereby self-contour resolution). A lone
    // sheet is undefined — sheets cut volumes (see the class @pre).
    if (tagged.size() == 1)
      config.mode = config.mode | tf::intersect_mode::within;
    _with_self = bool(config.mode & tf::intersect_mode::self_intersections);
    _intersections.build(tagged, config);
    auto &conv = _intersections.converter();

    auto apply_to_face = [&tagged](int tag, index_type object, const auto &f) {
      f(tagged[tag].faces()[object]);
    };
    auto get_mesh_point =
        [&tagged, &conv](int tag,
                         index_type id) -> tf::point<resolved_int_type, 3> {
      return conv.convert(
          tf::transformed(tagged[tag].points()[id], tf::frame_of(tagged[tag])));
    };

    _ig.build(_intersections, apply_to_face, get_mesh_point, config.mode,
              tf::exact::make_kernel(conv, config.tolerance));
    _fc.build(_ig, apply_to_face, get_mesh_point);
    _ag.build(_ig, _fc, tagged);

    auto get_point = [this, &get_mesh_point](
                         const auto &v,
                         index_type tag) -> tf::point<resolved_int_type, 3> {
      if (v.source == tf::intersect::graph::vertex_source::created)
        return _ig.points()[v.id];
      return get_mesh_point(int(tag), v.id);
    };

    _desc = tf::cut::make_arrangement_descriptor<resolved_int_type>(
        _ag, _fc, get_point, apply_to_face, _is_sheet);
    _inc = tf::cut::compute_domain_inclusions(_ag, _fc, _desc);
    _domain_volumes = tf::csg::graph::compute_arrangement_domain_volumes(
        tagged, _ag, _fc, _desc, get_point);
    auto seeds = tf::csg::graph::seed_inclusion_bits(
        _inc, _desc, _ag, _fc, _ig, tagged, conv, _domain_volumes,
        _domain_nesting_merges, _is_sheet);
    tf::cut::propagate_inclusion_bits(_inc, _desc, _ag, _fc, seeds);
    tf::cut::anchor_sheet_sides(_inc, _desc, _is_sheet);

    tf::buffer<tf::point<resolved_int_type, 3>> extra;
    if (tri == tf::triangulation_type::refined_cdt)
      _tris.build_refined(_ag, _fc, _ig, tagged, conv, extra);
    else
      _tris.build_stock(_ag, _fc, _ig, tagged, conv);
    auto ig_pts = _ig.points();
    _created_points.allocate(std::size_t(ig_pts.size()) + extra.size());
    tf::parallel_for_each(
        tf::make_sequence_range(std::size_t(ig_pts.size())),
        [&](std::size_t i) { _created_points[i] = ig_pts[index_type(i)]; });
    tf::parallel_copy(tf::make_range(extra),
                      tf::make_range(_created_points.begin() +
                                         std::size_t(ig_pts.size()),
                                     _created_points.end()));
  }

  /// @brief Pre-computed per-loop triangulation store (triangle corners are
  ///        loop vertices; folded coplanar-stack loops alias their
  ///        representative's range). Read by the csg extractions.
  auto loop_triangulations() const
      -> const tf::csg::graph::loop_triangulations<index_type,
                                                   resolved_int_type> & {
    return _tris;
  }

  /// @brief Unified created-points table (int lattice): the intersection
  ///        graph's points followed by any refinement-added points (splits,
  ///        then steiner). Created vertex ids index this buffer directly;
  ///        provenance of ids past the intersection graph's lives on
  ///        @ref loop_triangulations (`created_origin`).
  auto created_points() const
      -> const tf::buffer<tf::point<resolved_int_type, 3>> & {
    return _created_points;
  }

  /// @brief Per-form sheet mask (empty when no sheets were declared).
  auto is_sheet() const -> const tf::buffer<char> & { return _is_sheet; }

  /// True when the build carried @ref tf::intersect_mode::within —
  /// operands may self-overlap, so parity bits cannot be trusted for
  /// domain extraction (see @ref tf::make_csg_domains).
  auto with_self() const -> bool { return _with_self; }

  /// @brief Return the tagged forms view. If `Structs == none_t`,
  ///        returns the user's forms unchanged. Otherwise builds the
  ///        same `mapped_range(zip(_forms, _structs))` shape that
  ///        @ref tf::cut::dispatch::arrangement produces.
  auto forms() const {
    if constexpr (std::is_same_v<Structs, tf::none_t>) {
      return _forms;
    } else {
      return tf::make_mapped_range(tf::zip(_forms, _structs), [](auto pair) {
        auto &&[form, s] = pair;
        return std::apply(
            [&form = form](const auto &...structs) {
              return (form | ... | tf::tag(structs));
            },
            s);
      });
    }
  }

  auto converter() const -> const
      tf::exact::vertex_converter<resolved_int_type, pipeline_real_type, 3> & {
    return _intersections.converter();
  }
  auto intersection_graph() const
      -> const tf::intersection_graph<index_type, resolved_int_type> & {
    return _ig;
  }
  auto face_cuts() const
      -> const tf::face_cuts<index_type, resolved_int_type> & {
    return _fc;
  }
  auto arrangement() const -> const tf::arrangement_graph<index_type> & {
    return _ag;
  }
  auto descriptor() const
      -> const tf::cut::arrangement_descriptor<index_type> & {
    return _desc;
  }
  auto inclusion() const -> const tf::cut::domain_inclusions & { return _inc; }

  /// @brief Exact signed volume (2x, lattice units) per arrangement
  /// domain — the seeder's oracle. The most negative entry is the
  /// unbounded universe.
  auto domain_volumes() const
      -> const tf::buffer<typename tf::exact::meta<resolved_int_type>::T2> & {
    return _domain_volumes;
  }

  /// @brief `domain_of_side` merge pairs that repair contact-free nested
  /// shells (from the seeding cast). Consumed by `make_csg_domains`;
  /// irrelevant to the boolean mesh read. Empty when no nesting.
  auto domain_nesting_merges() const
      -> const tf::buffer<std::array<index_type, 2>> & {
    return _domain_nesting_merges;
  }

private:
  Forms _forms;
  tf::small_vector<Structs, 10> _structs;
  tf::buffer<char> _is_sheet;
  bool _with_self = false;
  tf::polygon_intersections<index_type, pipeline_real_type,
                            resolved_int_type>
      _intersections;
  tf::intersection_graph<index_type, resolved_int_type> _ig;
  tf::face_cuts<index_type, resolved_int_type> _fc;
  tf::arrangement_graph<index_type> _ag;
  tf::csg::graph::loop_triangulations<index_type, resolved_int_type> _tris;
  tf::buffer<tf::point<resolved_int_type, 3>> _created_points;
  tf::cut::arrangement_descriptor<index_type> _desc;
  tf::cut::domain_inclusions _inc;
  tf::buffer<typename tf::exact::meta<resolved_int_type>::T2> _domain_volumes;
  tf::buffer<std::array<index_type, 2>> _domain_nesting_merges;
};

} // namespace tf
