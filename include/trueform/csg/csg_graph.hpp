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
#include "../core/buffer.hpp"
#include "../core/frame_of.hpp"
#include "../core/none.hpp"
#include "../core/point.hpp"
#include "../core/small_vector.hpp"
#include "../core/transformed.hpp"
#include "../cut/arrangement_graph.hpp"
#include "../cut/arrangements/anchor_sheet_sides.hpp"
#include "../cut/arrangements/arrangement_descriptor.hpp"
#include "../cut/arrangements/component_labels.hpp"
#include "../cut/arrangements/compute_domain_inclusions.hpp"
#include "../cut/arrangements/make_arrangement_descriptor.hpp"
#include "../cut/arrangements/propagate_inclusion_bits.hpp"
#include "../cut/arrangement_config.hpp"
#include "./graph/compute_arrangement_domain_volumes.hpp"
#include "./graph/seed_inclusion_bits.hpp"
#include <array>
#include <utility>

namespace tf {

/// @ingroup csg
/// @brief Implicit N-form CSG graph — an @ref tf::arrangement_graph
///        plus classification: descriptor, domain inclusions, volumes,
///        and sheet anchoring.
///
/// Built once per form set, reused across many @ref tf::make_csg_mesh
/// calls (one per boolean expression).
///
/// Template parameters:
/// - `Forms`, `Structs` — deduced from the constructor.
/// - `Int` — exact-integer override (defaulted to `tf::none_t`,
///   resolved via @ref tf::exact::resolve_int_type from the form's
///   coordinate type).
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
  using arrangement_type = tf::arrangement_graph<
      tf::cut::arrangement_range_policy<Forms, Structs>, Int>;
  using forms_type = Forms;
  using structs_type = Structs;
  using index_type = typename arrangement_type::index_type;
  using input_real_type = typename arrangement_type::input_real_type;
  using resolved_int_type = typename arrangement_type::resolved_int_type;
  using pipeline_real_type = typename arrangement_type::pipeline_real_type;

  csg_graph(Forms forms, tf::small_vector<Structs, 10> structs,
            tf::arrangement_config config = {}, tf::buffer<char> is_sheet = {})
      : _is_sheet(std::move(is_sheet)),
        _arr(tf::cut::arrangement_range_policy<Forms, Structs>(
                 std::move(forms), std::move(structs)),
             config) {
    auto tagged = _arr.forms();
    auto &conv = _arr.converter();
    const auto &rt = _arr.triangulations();
    const auto &created_pts = _arr.created_points();
    _labels.build(_arr.face_regions(), tagged, _arr.coplanar_pairs(),
                  _arr.dead_loops(),
                  index_type(created_pts.size()));
    _labels.bind_exposed(rt.exposed_ranges(), rt.loops().size());
    const auto &ag = _labels;

    auto apply_to_face = [&tagged](int tag, index_type object, const auto &f) {
      f(tagged[tag].faces()[object]);
    };
    auto get_mesh_point =
        [&tagged, &conv](int tag,
                         index_type id) -> tf::point<resolved_int_type, 3> {
      return conv.convert(
          tf::transformed(tagged[tag].points()[id], tf::frame_of(tagged[tag])));
    };
    auto get_point = [&created_pts, &get_mesh_point](
                         const auto &v,
                         index_type tag) -> tf::point<resolved_int_type, 3> {
      if (v.source == tf::intersect::graph::vertex_source::created)
        return created_pts[std::size_t(v.id)];
      return get_mesh_point(int(tag), v.id);
    };

    _desc = tf::cut::make_arrangement_descriptor<resolved_int_type>(
        ag, _arr.face_regions(), get_point, apply_to_face, _is_sheet);
    _inc = tf::cut::compute_domain_inclusions(ag, _arr.face_regions(), _desc);
    _domain_volumes = tf::csg::graph::compute_arrangement_domain_volumes(
        tagged, ag, rt, _desc, get_point);
    auto seeds = tf::csg::graph::seed_inclusion_bits(
        _inc, _desc, ag, rt, created_pts, tagged, conv, _domain_volumes,
        _domain_nesting_merges, _is_sheet);
    tf::cut::propagate_inclusion_bits(_inc, _desc, ag, _arr.face_regions(), seeds);
    // Sheets coplanar-folded into another wall have no fragments of
    // their own to anchor from, and their winding seeds are degenerate
    // (evaluated exactly on the shared wall): anchor them through the
    // carrying component, mirrored by the fold's reversed flag.
    {
      auto loop_labels = ag.loop_labels();
      auto descs = _arr.face_regions().descriptors();
      for (const auto &p : ag.coplanar_pairs()) {
        // Pairs form the full clique of a coincident stack, so a dead
        // region also appears as a "survivor"; the direct (live, dead)
        // pair always exists — skip the dead-survivor ones. A survivor
        // can clique with several same-tag dead regions; the
        // consecutive-duplicate guard keeps one entry per fold.
        const auto c = index_type(loop_labels[p[0]]);
        if (c == tf::cut::component_labels<index_type>::none_label)
          continue;
        const auto t = index_type(descs[p[1]].tag);
        if (t < index_type(_is_sheet.size()) && _is_sheet[t]) {
          const std::array<index_type, 3> fold{c, t, p[2]};
          if (_sheet_folds.size() == 0 ||
              _sheet_folds[_sheet_folds.size() - 1] != fold)
            _sheet_folds.push_back(fold);
        }
      }
    }
    tf::cut::anchor_sheet_sides(_inc, _desc, _is_sheet, _sheet_folds);
  }

  /// @brief The arrangement this graph classifies.
  auto arrangement() const -> const arrangement_type & { return _arr; }

  /// @brief The classification label tier over the arrangement.
  auto labels() const -> const tf::cut::component_labels<index_type> & {
    return _labels;
  }

  auto forms() const { return _arr.forms(); }
  auto converter() const -> decltype(auto) { return _arr.converter(); }
  auto intersection_graph() const -> decltype(auto) {
    return _arr.intersection_graph();
  }
  auto triangulations() const -> decltype(auto) {
    return _arr.triangulations();
  }
  auto created_points() const -> decltype(auto) {
    return _arr.created_points();
  }
  auto with_self() const -> bool { return _arr.with_self(); }

  /// @brief Per-form sheet mask (empty when no sheets were declared).
  auto is_sheet() const -> const tf::buffer<char> & { return _is_sheet; }

  /// Sheets coplanar-folded into another component's wall:
  /// `(component, sheet tag, reversed)` per fold.
  auto sheet_folds() const -> const tf::buffer<std::array<index_type, 3>> & {
    return _sheet_folds;
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
  tf::buffer<char> _is_sheet;
  arrangement_type _arr;
  tf::cut::component_labels<index_type> _labels;
  tf::buffer<std::array<index_type, 3>> _sheet_folds;
  tf::cut::arrangement_descriptor<index_type> _desc;
  tf::cut::domain_inclusions _inc;
  tf::buffer<typename tf::exact::meta<resolved_int_type>::T2> _domain_volumes;
  tf::buffer<std::array<index_type, 2>> _domain_nesting_merges;
};

} // namespace tf
