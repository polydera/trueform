/*
 * Copyright (c) 2025 XLAB
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

#include "../arrangement/graph_builders.hpp"
#include "../intersect/config_validation.hpp"
#include "trueform/arrangement/arrangement_config.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/csg/outer_shell.hpp"
#include "trueform/csg/make_outer_shell.hpp"

#include <cstddef>

namespace tf::cpp {

template <typename Index, typename Real, std::size_t Ngon>
auto outer_shell(const mesh<Index, Real, 3, Ngon> &value,
                 tf::intersect_config intersect_config)
    -> tf::polygons_buffer<Index, Real, 3, Ngon> {
  intersect_detail::require_config<Real>(intersect_config, "outer_shell", true);
  value.require_indices();
  if (value.number_of_faces() == 0)
    return {};

  // The shell is the operand's own boundary, so it is read where the operand
  // was authored rather than where a placement puts it. The view is bound,
  // because the graph holds the form taken from it.
  const auto operand = value.at_identity();
  auto graph = graph_builders::build_self_csg_graph(
      operand.topology_form(), tf::arrangement_config{intersect_config});
  return tf::make_outer_shell<Real>(graph);
}

} // namespace tf::cpp
