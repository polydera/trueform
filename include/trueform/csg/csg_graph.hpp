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
#include "../arrangement/arrangement_graph.hpp"
#include "../core/buffer.hpp"
#include "../core/frame_of.hpp"
#include "../core/none.hpp"
#include "../core/point.hpp"
#include "../core/transformed.hpp"
#include "./graph/assert_nesting_merges_keep_closed_bits.hpp"
#include "./graph/arrangement_descriptor.hpp"
#include "./graph/component_crossings.hpp"
#include "./graph/compute_arrangement_domain_volumes.hpp"
#include "./graph/domain_depths.hpp"
#include "./graph/domain_inclusions.hpp"
#include "./graph/make_arrangement_descriptor.hpp"
#include "./graph/make_closed_volume_tags.hpp"
#include "./graph/make_domain_components.hpp"
#include "./graph/make_sheet_sides.hpp"
#include "./graph/mark_nested_sheet_wraps.hpp"
#include "./graph/mark_sheet_wraps.hpp"
#include "./graph/propagate_inclusion_bits.hpp"
#include "./graph/propagate_sheet_sides.hpp"
#include "./graph/seed_inclusion_bits.hpp"
#include "./graph/triangle_component_labels.hpp"
#include <array>
#include <cstddef>
#include <utility>

namespace tf {

/// @ingroup csg
/// @brief Implicit N-form CSG graph — an @ref tf::arrangement_graph
///        plus classification: descriptor, domain inclusions, volumes,
///        and the sheets' sides.
///
/// Built once per form set, reused across many @ref tf::make_csg_mesh
/// calls (one per boolean expression).
///
/// Template parameters:
/// - `Policy` — the arrangement storage policy (homogeneous range or
///   heterogeneous pair), exactly as on @ref tf::arrangement_graph.
/// - `Int` — exact-integer override (defaulted to `tf::none_t`,
///   resolved via @ref tf::exact::resolve_int_type from the form's
///   coordinate type).
/// - `Arrangement` — the arrangement class template the classification
///   tier sits on; the factory that built the arrangement fixes it.
///
/// Output coordinate type is not on the graph — materialisation
/// happens in @ref tf::make_csg_mesh, which takes its own
/// `OutputCoordinateType` parameter.
///
/// @pre A one-form graph's form is a volume, not a declared sheet — a
///      lone sheet is a cutter with nothing to cut.
template <typename Policy, typename Int = tf::none_t,
          template <typename, typename> class Arrangement =
              tf::arrangement_graph>
class csg_graph {
public:
  using arrangement_type = Arrangement<Policy, Int>;
  using policy_type = Policy;
  using index_type = typename arrangement_type::index_type;
  using input_real_type = typename arrangement_type::input_real_type;
  using resolved_int_type = typename arrangement_type::resolved_int_type;
  using pipeline_real_type = typename arrangement_type::pipeline_real_type;
  static constexpr std::size_t face_static_size =
      arrangement_type::face_static_size;

  /// Takes the arrangement already built — every operand shape reaches
  /// it through @ref tf::make_arrangement_graph, which is the one place
  /// that completes a form's missing structures.
  csg_graph(arrangement_type arr, tf::buffer<char> is_sheet = {})
      : _is_sheet(std::move(is_sheet)), _arr(std::move(arr)) {
    auto apply_form = _arr.apply_to_form();
    auto &conv = _arr.converter();
    _labels.build(_arr, apply_form);

    auto apply_to_face = [apply_form](int tag, index_type object,
                                      const auto &f) {
      apply_form(index_type(tag),
                 [&](const auto &form) { f(form.faces()[object]); });
    };
    // the arrangement's own reader, never a second one: where an
    // original vertex stands is stated once, above both tiers
    auto get_mesh_point = _arr.lattice().reader(apply_form);

    _desc = tf::csg::graph::make_arrangement_descriptor<resolved_int_type>(
        _arr, _labels, get_mesh_point, apply_to_face, _is_sheet);
    _inc = tf::csg::graph::make_domain_inclusions(_arr.n_tags(),
                                                  _desc.n_domains);
    _domain_volumes =
        tf::csg::graph::compute_arrangement_domain_volumes<index_type,
                                                           resolved_int_type>(
            _arr, _labels, _desc, apply_form, get_mesh_point);
    _crossings =
        tf::csg::graph::make_component_crossings<index_type>(_arr, _labels);
    // The depths and the sheet sides are the floods' working state and the
    // inclusion bits their published answer, so they live exactly from the
    // seeding cast to the publication that ends the floods.
    {
      auto depths = tf::csg::graph::make_domain_depths(_arr.n_tags(),
                                                       _desc.n_domains);
      tf::buffer<std::array<index_type, 4>> sheet_seeds;
      tf::buffer<std::array<index_type, 2>> rim_wraps;
      auto seeds = tf::csg::graph::seed_inclusion_bits<index_type,
                                                       resolved_int_type>(
          depths, _desc, _arr, _labels, apply_form, conv, get_mesh_point,
          _domain_volumes, _domain_nesting_merges, sheet_seeds, rim_wraps,
          _is_sheet);
      auto domain_components = tf::csg::graph::make_domain_components(_desc);
      tf::csg::graph::propagate_inclusion_bits(
          _inc, depths, _desc, _crossings, domain_components, _is_sheet, seeds);
      auto sides = tf::csg::graph::make_sheet_sides(_desc, _crossings,
                                                    _arr.n_tags(), _is_sheet);
      tf::csg::graph::mark_sheet_wraps(sides, _desc, _crossings,
                                       _labels.triangle_labels(),
                                       _arr.exposed_of_row());
      tf::csg::graph::mark_nested_sheet_wraps(sides, rim_wraps,
                                              _domain_nesting_merges);
      tf::csg::graph::propagate_sheet_sides(_inc, sides, sheet_seeds, _desc,
                                            _crossings, domain_components);
    }
#ifndef NDEBUG
    tf::csg::graph::assert_nesting_merges_keep_closed_bits(
        _domain_nesting_merges, _inc,
        tf::csg::graph::make_closed_volume_tags(_desc, _labels, _arr.n_tags(),
                                                _is_sheet));
#endif
  }

  /// @brief The arrangement this graph classifies.
  auto arrangement() const -> const arrangement_type & { return _arr; }

  /// @brief The arrangement carriers that hold no product — see
  ///        @ref tf::arrangement_graph::failed. Empty means this
  ///        classification stands on a complete arrangement.
  auto failed() const { return _arr.failed(); }

  /// @brief The classification label tier over the arrangement.
  auto labels() const
      -> const tf::csg::graph::triangle_component_labels<index_type> & {
    return _labels;
  }

  auto converter() const -> decltype(auto) { return _arr.converter(); }
  auto created_points() const -> decltype(auto) {
    return _arr.created_points();
  }
  auto with_self() const -> bool { return _arr.with_self(); }

  /// @brief Per-form sheet mask (empty when no sheets were declared).
  auto is_sheet() const -> const tf::buffer<char> & { return _is_sheet; }


  auto descriptor() const
      -> const tf::csg::graph::arrangement_descriptor<index_type> & {
    return _desc;
  }
  auto inclusion() const -> const tf::csg::graph::domain_inclusions & {
    return _inc;
  }

  /// @brief The layers every wall carries, and the components whose
  /// coincident stack is not one fact — `n_defects` of them, the first
  /// named by `defect_component`. A defect is reported, never refused:
  /// those components are classified on their representative's stack.
  auto crossings() const
      -> const tf::csg::graph::component_crossings<index_type> & {
    return _crossings;
  }

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
  tf::csg::graph::triangle_component_labels<index_type> _labels;
  tf::csg::graph::arrangement_descriptor<index_type> _desc;
  tf::csg::graph::component_crossings<index_type> _crossings;
  tf::csg::graph::domain_inclusions _inc;
  tf::buffer<typename tf::exact::meta<resolved_int_type>::T2> _domain_volumes;
  tf::buffer<std::array<index_type, 2>> _domain_nesting_merges;
};

} // namespace tf
