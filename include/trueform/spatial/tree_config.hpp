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

namespace tf {

/// @ingroup spatial_configuration
/// @brief Configuration object for controlling tree construction.
///
/// `tree_config` defines the parameters used during tree construction,
/// including maximum inner node and leaf node sizes.
///
/// Users can create this directly or use the @ref tf::config_tree helper
/// function for convenience.
struct tree_config {
  int inner_size;
  int leaf_size;
};

/// @ingroup spatial_configuration
/// @brief Create a tree configuration with specified node sizes.
///
/// Returns a `tree_config` object that defines how the tree will be built.
///
/// @param inner_size The maximum number of children per internal node.
/// @param leaf_size The maximum number of primitives per leaf node.
///
/// @return A `tree_config` instance for use with `tf::tree::build(...)`.
inline auto config_tree(int inner_size, int leaf_size) -> tree_config {
  return tree_config{inner_size, leaf_size};
}

} // namespace tf
