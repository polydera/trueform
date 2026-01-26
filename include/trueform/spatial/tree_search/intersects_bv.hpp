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
#include "../../core/intersects.hpp"
#include "../mod_tree_like.hpp"
#include "../tree_like.hpp"

namespace tf::spatial {

template <typename TreePolicy, typename Geometry>
auto intersects_bv(const tf::tree_like<TreePolicy> &tree,
                   const Geometry &geometry) -> bool {
  if (!tree.nodes().size())
    return false;
  return tf::intersects(tree.nodes()[0].bv, geometry);
}

template <typename ModTreePolicy, typename Geometry>
auto intersects_bv(const tf::mod_tree_like<ModTreePolicy> &tree,
                   const Geometry &geometry) -> bool {
  if (tree.main_tree().nodes().size() &&
      tf::intersects(tree.main_tree().nodes()[0].bv, geometry))
    return true;
  if (tree.delta_tree().nodes().size() &&
      tf::intersects(tree.delta_tree().nodes()[0].bv, geometry))
    return true;
  return false;
}

} // namespace tf::spatial
