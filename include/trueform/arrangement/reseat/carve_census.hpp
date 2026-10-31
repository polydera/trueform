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
#include "../../core/reallocate.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/find_moved_identity_planes.hpp"
#include "../../intersect/intersect_config.hpp"
#include "./make_carve_package.hpp"
#include "./reseat_census.hpp"
#include "./reseat_pieces.hpp"
#include "./reseat_refusal.hpp"
#include "./state_reseat_targets.hpp"
#include <algorithm>
#include <array>
#include <iterator>
#include <utility>

namespace tf::arrangement::reseat {

/// The census's statement carved into the built `world` once its welds
/// stand there (@ref
/// tf::intersect::graph::local_arrangement::restate_census). `rows` are
/// `{kind, source, target}` in the merge currency, a target past the
/// world's created identities naming `minted[target - n_created_points]`.
/// A kind-1 row stands in the world already, and only the carriers it
/// leaves are carved; a kind-0 row retires its original with the carve, so
/// every face holding the original enters and every plane holding it
/// restates. Every source face enters as it stands (`standing`). The fixed
/// point is stated with its maps and moved rows (`graph.state_reseat`),
/// which stay empty when nothing is stated, and `touched_planes` carries the
/// statement's own dirty set — the planes the carve restates and every plane
/// holding a group the restatement touched; a carve that cannot be stated
/// returns false — it is refused in the census and the world stands as
/// built.
template <typename Index, typename Int, typename World, typename ApplyToForm,
          typename Reader, typename Converter, typename Graph>
auto carve_census(const World &world, const standing_pieces<Index> &standing,
                  const tf::buffer<tf::exact::pt3<Int>> &minted,
                  const tf::buffer<std::array<Index, 3>> &rows,
                  const ApplyToForm &apply_to_form, const Reader &reader,
                  const Converter &converter, Index n_tags,
                  tf::intersect_config config, Graph &graph,
                  tf::buffer<Index> &row_images,
                  tf::buffer<Index> &canon_images,
                  tf::buffer<Index> &touched_planes,
                  tf::buffer<std::array<Index, 3>> &moved_rows,
                  tf::buffer<Index> &dying_faces) -> bool {
  if (standing.foreign) {
    graph.policy().refuse_carve(reseat_refusal::name_foreign,
                                carve_reach_census{});
    return false;
  }
  const auto &identities = graph.identities();
  const auto n_created = world.n_created_points(identities);
  if constexpr (tf::core::verification)
    for (const auto &row : rows) {
      tf::core::verify(row[0] == Index(0) || identities.local_target(
                                                 Index(1), row[1]) != Index(-1),
                       "a census kind-1 row stands in the world");
      tf::core::verify(row[2] < n_created + Index(minted.size()),
                       "a census row names this world or its own mints");
    }
  tf::buffer<std::array<Index, 2>> retiring;
  for (const auto &row : rows)
    if (row[0] == Index(0))
      retiring.push_back({Index(0), row[1]});
  std::sort(retiring.begin(), retiring.end());
  retiring.erase_till_end(std::unique(retiring.begin(), retiring.end()));

  const auto quantum = graph.quantum();
  const auto created =
      tf::make_mapped_range(tf::make_sequence_range(n_created), [&](auto id) {
        return world.point_of(identities, id, reader);
      });
  const auto targets = state_reseat_targets<Index, Int>(
      world, rows, created, standing, apply_to_form, reader, quantum);
  if (targets.size() == 0 && retiring.size() == 0 && minted.size() == 0)
    return true;
  const auto retiring_planes = tf::intersect::graph::find_moved_identity_planes(
      world.tables(), world.vertex_offsets(), retiring,
      [&world](Index face) { return world.plane_of_face(face); });
  // the census's mints follow the carve's own, and its retirements join the
  // carve's welds, their planes the carve's
  const auto complete = [&](reseat_package<Index, Int> &package) {
    const auto rebase = Index(package.minted.size());
    tf::core::append(minted, package.minted);
    for (const auto &row : rows)
      if (row[0] == Index(0))
        package.welds.push_back(
            {Index(0), row[1], row[2] < n_created ? row[2] : row[2] + rebase});
    tf::buffer<Index> planes;
    std::set_union(package.planes.begin(), package.planes.end(),
                   retiring_planes.begin(), retiring_planes.end(),
                   std::back_inserter(planes));
    package.planes = std::move(planes);
  };
  // a face the plane arrangement promoted is already a cut-world citizen:
  // the entrance never names it twice, and the weld wave re-reads its rows
  const auto promoted = graph.refine_promoted();
  const auto pa_named = [&promoted](Index tag, Index object) {
    return std::binary_search(promoted.begin(), promoted.end(),
                              std::array<Index, 2>{tag, object});
  };
  carve_reach_census reach;
  const auto package = make_carve_package<Index, Int>(
      world, graph.identities(), standing, created, targets, apply_to_form,
      reader, converter, n_tags, quantum, config, complete, pa_named, reach);
  if (!package.ok) {
    graph.policy().refuse_carve(package.refusal, reach);
    return false;
  }
  tf::buffer<Index> restated_planes;
  graph.state_reseat(package, row_images, canon_images, restated_planes,
                     moved_rows);
  graph.policy().count_carve(reach);
  dying_faces.allocate(package.retired.size() + package.restated.size());
  std::merge(package.retired.begin(), package.retired.end(),
             package.restated.begin(), package.restated.end(),
             dying_faces.begin());
  std::set_union(package.planes.begin(), package.planes.end(),
                 restated_planes.begin(), restated_planes.end(),
                 std::back_inserter(touched_planes));
  return true;
}

} // namespace tf::arrangement::reseat
