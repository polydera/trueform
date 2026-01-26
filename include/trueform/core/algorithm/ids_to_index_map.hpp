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

#include "../index_map.hpp"
#include "../views/indirect_range.hpp"
#include "../views/sequence_range.hpp"
#include "./parallel_copy.hpp"
#include "./parallel_fill.hpp"

namespace tf {

/// @ingroup core_algorithms
/// @brief Create an index map from a list of IDs.
///
/// Builds a mapping from original IDs to compact sequential indices.
///
/// @tparam Index The index type.
/// @tparam Range The ID range type.
/// @param ids The list of kept IDs.
/// @param mapping Output index map buffer.
/// @param total_elements Total number of elements.
template <typename Index, typename Range>
auto ids_to_index_map(const Range &ids, tf::index_map_buffer<Index> &mapping,
                      Index total_elements) {
  mapping.f().allocate(total_elements);
  tf::parallel_fill(mapping.f(), total_elements);

  mapping.kept_ids().allocate(ids.size());
  tf::parallel_copy(ids, mapping.kept_ids());

  tf::parallel_copy(tf::make_sequence_range(mapping.kept_ids().size()),
                    tf::make_indirect_range(mapping.kept_ids(), mapping.f()));
}

/// @ingroup core_algorithms
/// @brief Create an index map with custom offset and empty tag.
/// @overload
template <typename Index, typename Range>
auto ids_to_index_map(const Range &ids, tf::index_map_buffer<Index> &mapping,
                      Index total_elements, Index offset, Index empty_tag) {
  mapping.f().allocate(total_elements);
  tf::parallel_fill(mapping.f(), empty_tag);

  mapping.kept_ids().allocate(ids.size());
  tf::parallel_copy(ids, mapping.kept_ids());

  tf::parallel_copy(
      tf::make_sequence_range(offset, Index(mapping.kept_ids().size() + offset)),
      tf::make_indirect_range(mapping.kept_ids(), mapping.f()));
}

/// @ingroup core_algorithms
/// @brief Create and return an index map from IDs.
/// @overload
template <typename Index, typename Range>
auto ids_to_index_map(const Range &ids, Index total_elements) {
  tf::index_map_buffer<Index> mapping;
  tf::ids_to_index_map<Index>(ids, mapping, total_elements);
  return mapping;
}

} // namespace tf
