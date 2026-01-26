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

/// @ingroup topology_types
/// @brief A hierarchical identifier with a main ID and a sub-ID.
///
/// Used to identify elements within a parent element, such as an edge
/// within a face (face_id, edge_id) or a vertex within a face.
///
/// @tparam Index The type for the main identifier.
/// @tparam SubIndex The type for the sub-identifier.
template <typename Index, typename SubIndex> struct scoped_id {
  Index id;       ///< The main identifier (e.g., face index).
  SubIndex sub_id;  ///< The sub-identifier within the parent (e.g., edge index).
};

/// @ingroup topology_types
/// @brief Creates a scoped_id from a main ID and sub-ID.
///
/// @tparam Index The type for the main identifier.
/// @tparam SubIndex The type for the sub-identifier.
/// @param id The main identifier.
/// @param sub_id The sub-identifier.
/// @return A scoped_id combining both.
template <typename Index, typename SubIndex>
auto make_scoped_id(const Index &id, const SubIndex &sub_id) {
  return scoped_id<Index, SubIndex>{id, sub_id};
}

} // namespace tf
