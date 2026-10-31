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
#include "../../core/config/verify.hpp"
#include "../../core/range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/intersect_config.hpp"
#include "../identity/identity_substrate.hpp"
#include "./find_retirement_entrants.hpp"
#include "./make_first_carve_round.hpp"
#include "./make_next_carve_round.hpp"
#include "./make_reseat_delta_forms.hpp"
#include "./make_reseat_delta_world.hpp"
#include "./make_reseat_package.hpp"
#include "./reseat_census.hpp"
#include "./reseat_pieces.hpp"
#include "./reseat_refusal.hpp"
#include "./reseat_targets.hpp"
#include "./state_reseat_pieces.hpp"
#include "./state_reseat_reach.hpp"
#include "./state_reseat_targets.hpp"
#include "./state_reseat_touched.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace tf::arrangement::reseat {

/// The rounds a carve may run, and the times its reach may widen, before it
/// is refused.
inline constexpr std::size_t carve_round_cap = 4;
inline constexpr std::size_t carve_widen_cap = 8;

/// The carve of `targets` into the built `world`, its identities read through
/// `identities`, run to its fixed point: each round's delta states the
/// targets the carved world would, a face the delta splits or a retirement
/// reaches widens the reach, and `complete`
/// adds to each package what is stated beside the carve before its
/// retirements are read. A source face enters as it stands (`standing`).
/// Returns the package to state, `ok`, or the refusal that leaves the world
/// as built; `reach` counts what the carve reached on the way.
template <typename Index, typename Int, typename World, typename Created,
          typename ApplyToForm, typename Reader, typename Converter,
          typename Complete, typename AlreadyNamed>
auto make_carve_package(
    const World &world,
    const tf::arrangement::identity_substrate<Index, Int> &identities,
    const standing_pieces<Index> &standing, const Created &created,
    const reseat_targets<Index, Int> &targets, const ApplyToForm &apply_to_form,
    const Reader &reader, const Converter &converter, Index n_tags, Int quantum,
    tf::intersect_config config, const Complete &complete,
    const AlreadyNamed &already_named, carve_reach_census &reach_census)
    -> reseat_package<Index, Int> {
  const world_face_lookup<Index, World> world_face_of(world);
  const standing_pieces<Index> no_standing;
  std::vector<carve_round<Index, Int>> rounds;
  if (targets.size() != 0)
    rounds.push_back(make_first_carve_round(targets, apply_to_form));
  tf::buffer<std::array<Index, 2>> forced;
  for (std::size_t round = 0, widen = 0;;) {
    reseat_package<Index, Int> package;
    std::size_t neighbours = 0;
    if (rounds.size() == 0 && forced.size() == 0) {
      package.ok = true;
      package.minted_base = world.n_created_points(identities);
    } else {
      const auto pieces = state_reseat_pieces<Index, Int>(
          rounds, standing, n_tags, apply_to_form, reader, created);
      if (pieces.pieces.size() == 0 && forced.size() == 0) {
        package.refusal = reseat_refusal::no_pieces;
        return package;
      }
      auto touched = state_reseat_touched<Index, Int>(
          pieces, n_tags, apply_to_form, reader, created, converter);
      std::sort(touched.begin(), touched.end());
      touched.erase_till_end(std::unique(touched.begin(), touched.end()));
      std::sort(forced.begin(), forced.end());
      forced.erase_till_end(std::unique(forced.begin(), forced.end()));
      const auto reach = state_reseat_reach<Index>(world, standing,
                                                   pieces.carved, forced,
                                                   touched);
      const auto forms = make_reseat_delta_forms<Index, Int>(
          pieces, reach, standing, n_tags, apply_to_form, reader, created);
      ++reach_census.delta_rounds;
      for (const auto &faces : forms.faces)
        reach_census.delta_faces += faces.size();
      const auto delta = make_reseat_delta_world<Index, Int>(forms, config);
      const auto delta_forms = delta.apply_to_form();
      const auto delta_reader = delta.reader();
      const auto delta_created = tf::make_mapped_range(
          tf::make_sequence_range(
              delta.world.n_created_points(delta.identities)),
          [&](auto id) {
            return delta.world.point_of(delta.identities, id, delta_reader);
          });
      const auto next = state_reseat_targets<Index, Int>(
          delta.world, tf::make_range(delta.identities.local_merges()),
          delta_created, no_standing, delta_forms, delta_reader, quantum);
      if (next.size() != 0) {
        if (++round == carve_round_cap) {
          package.refusal = reseat_refusal::rounds_capped;
          return package;
        }
        rounds.push_back(make_next_carve_round<Index, Int>(
            next, pieces, standing, forms, delta_forms));
        continue;
      }
      package = make_reseat_package<Index, Int>(world, identities, standing,
                                                pieces, reach, forms, delta,
                                                reader, apply_to_form, n_tags);
      neighbours = reach.neighbours.size();
    }
    if constexpr (tf::core::verification) {
      tf::core::verify(package.ok || package.refusal != reseat_refusal::none,
                       "a carve the package does not state names its refusal");
      tf::core::verify(package.refusal != reseat_refusal::reach_unbounded ||
                           package.promote.size() != 0,
                       "a package that widens its reach names the faces");
    }
    auto reached = package.promote;
    std::size_t foreign = 0;
    if (package.ok) {
      foreign = std::size_t(std::count_if(
          package.welds.begin(), package.welds.end(),
          [minted_base = package.minted_base](const std::array<Index, 3> &w) {
            return (w[0] == Index(0) || w[1] < minted_base) &&
                   w[2] < minted_base;
          }));
      complete(package);
      reached = find_retirement_entrants<Index>(world, world_face_of, package,
                                                apply_to_form, already_named);
    } else if (package.refusal != reseat_refusal::reach_unbounded)
      return package;
    if (reached.size() == 0) {
      reach_census.restated = package.restated.size();
      reach_census.neighbours = neighbours;
      reach_census.foreign_retirements = foreign;
      return package;
    }
    if (widen++ == carve_widen_cap) {
      package.ok = false;
      package.refusal = reseat_refusal::reach_unbounded;
      return package;
    }
    ++reach_census.widens;
    for (const auto &key : reached)
      forced.push_back(key);
  }
}

} // namespace tf::arrangement::reseat
