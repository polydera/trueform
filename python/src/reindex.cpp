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

#include "trueform/python/reindex.hpp"

namespace tf::py {

auto register_reindex(nanobind::module_ &m) -> void {
  // Create reindex submodule
  auto reindex_module = m.def_submodule("reindex", "Reindexing operations");

  // Register reindex components to submodule
  register_reindex_reindex_by_ids(reindex_module);
  register_reindex_reindex_by_mask(reindex_module);
  register_reindex_split_into_components(reindex_module);
}

} // namespace tf::py
