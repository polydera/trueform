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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/meta.hpp"
#include "./plane_census.hpp"
#include "./plane_graph.hpp"
#include "./plane_identity_birth.hpp"
#include "./plane_identity_names.hpp"
#include <cstddef>

namespace tf::intersect::graph {

/// THE CLOSURE — and it is a sort. Names are the identities, so the runs
/// of equal names are the classes and nothing decides sameness
/// numerically. Each class is then BORN once: its exact rational
/// parameter on its canonical smallest root — transient, inside this
/// expression, never stored — rounded to a dyadic and blended into a
/// lattice position on that root. A class that names an existing point is
/// not born at all; it IS that point.
///
/// The return value is the class count: `class_id` holds the identity a
/// class NAMES (`-1` when it will mint one), `class_root` and `class_t`
/// the root a class was born on and its dyadic parameter there.
template <typename Index, typename Int, typename GetPoint>
auto close_plane_classes(
    const plane_graph<Index, Int> &g, const GetPoint &get_point,
    tf::buffer<plane_name_statement<Index>> &statements,
    tf::buffer<Index> &class_offsets, tf::buffer<Index> &class_id,
    tf::buffer<Index> &class_root,
    tf::buffer<typename tf::exact::meta<Int>::param_type> &class_t,
    plane_census &census) -> Index {
  using param_t = typename tf::exact::meta<Int>::param_type;
  if (statements.size() == 0)
    return 0;
  const auto n_classes = close_plane_names<Index>(statements, class_offsets);
  census.classes = std::size_t(n_classes);
  class_id.allocate(std::size_t(n_classes));
  class_root.allocate(std::size_t(n_classes));
  class_t.allocate(std::size_t(n_classes));
  tf::parallel_for_each(
      tf::make_sequence_range(n_classes),
      [&](Index cls) {
        const auto &statement =
            statements[std::size_t(class_offsets[std::size_t(cls)])];
        class_root[std::size_t(cls)] = Index(-1);
        class_t[std::size_t(cls)] = param_t(0);
        if (statement.name[0] == Index(0)) {
          class_id[std::size_t(cls)] = statement.name[1];
          return;
        }
        class_id[std::size_t(cls)] = Index(-1);
        if (statement.name[0] == Index(1))
          return;
        class_root[std::size_t(cls)] = statement.root;
        class_t[std::size_t(cls)] =
            birth_plane_point<Index, Int>(g, get_point, statement);
      },
      tf::checked);
  return n_classes;
}

} // namespace tf::intersect::graph
