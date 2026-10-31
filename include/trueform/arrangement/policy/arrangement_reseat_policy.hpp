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
#include "../../exact/resolve_int_type.hpp"
#include "../../intersect/intersect_config.hpp"
#include "../construct/arrangement_map_data.hpp"
#include "../reseat/reseat_census.hpp"
#include "../reseat/reseat_refusal.hpp"
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::arrangement {

/// @brief The storage policy every arrangement graph is built on: the
///        caller's operands and what the re-seat stated into the build.
template <typename Base, typename Int> struct arrangement_reseat_policy {
  using index_type = typename Base::index_type;
  using input_real_type = typename Base::input_real_type;
  static constexpr std::size_t face_static_size = Base::face_static_size;
  static constexpr std::size_t static_n_tags = Base::static_n_tags;
  /// Whether the operands' arity admits a carve.
  static constexpr bool reseats = face_static_size == 3;

  explicit arrangement_reseat_policy(Base base) : _base(std::move(base)) {}

  auto n_tags() const -> index_type { return _base.n_tags(); }
  auto reseat_census() const -> const tf::arrangement::reseat::reseat_census & {
    return _census;
  }

  auto make_apply_to_form() const { return _base.make_apply_to_form(); }

  template <typename I> auto make_lattice(double tolerance) const {
    static_assert(std::is_same_v<I, Int>,
                  "the lattice is the one the policy builds on");
    return _base.template make_lattice<Int>(tolerance);
  }

  template <typename Ibp, typename Lattice>
  auto build_intersections(Ibp &ibp, const Lattice &lattice,
                           tf::intersect_config config) const {
    _base.build_intersections(ibp, lattice, config);
  }

  template <typename Index, typename Triangles>
  auto concatenated_output_faces(
      const tf::arrangement::arrangement_map_data<Index> &map_data,
      const Triangles &triangles) const {
    return _base.template concatenated_output_faces<Index>(map_data, triangles);
  }

  auto count_carve(const tf::arrangement::reseat::carve_reach_census &reach)
      -> void {
    ++_census.carves;
    _census.reach += reach;
  }
  auto refuse_carve(tf::arrangement::reseat::reseat_refusal refusal,
                    const tf::arrangement::reseat::carve_reach_census &reach)
      -> void {
    _census.refusal = refusal;
    _census.reach += reach;
  }

private:
  Base _base;
  tf::arrangement::reseat::reseat_census _census;
};

/// The policy a graph over `Base` is built on, its lattice the integer the
/// request `Int` resolves to.
template <typename Base, typename Int>
using reseat_policy_t = arrangement_reseat_policy<
    Base, tf::exact::resolve_int_type<Int, typename Base::input_real_type>>;

} // namespace tf::arrangement
