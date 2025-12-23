#pragma once
#include "../index_map.hpp"
#include "./invert_map_with_nones.hpp"
#include "./mask_to_map.hpp"
namespace tf {

/// @ingroup core_algorithms
/// @brief Constructs a bidirectional mapping from a boolean mask.
///
/// This function populates an existing @ref tf::mapping_val<Index> with a
/// forward mapping from original indices to new compacted indices, and an
/// inverse mapping from the new compacted indices back to the original ones.
/// Only entries marked as `true` in the mask are assigned a compacted index.
///
/// The forward mapping is computed using @ref tf::mask_to_map, and the inverse
/// is filled via
/// @ref tf::invert_map_with_nones. Entries not included in the compacted set
/// are set to a sentinel value (equal to `mask.size()`).
///
/// @tparam Range A range with boolean values indicating which entries to keep.
/// @tparam Index The integer type used for indexing.
/// @param mask The boolean mask indicating which entries to retain.
/// @param mapping The output bidirectional mapping that will be populated.
///
/// @see tf::mask_to_map
/// @see tf::invert_map_with_nones
template <typename Range, typename Index>
auto mask_to_index_map(const Range &mask,
                       tf::index_map_buffer<Index> &mapping) {
  mapping.f().allocate(mask.size());
  auto n_kept = tf::mask_to_map(mask, mapping.f());
  mapping.kept_ids().allocate(n_kept);
  tf::invert_map_with_nones(mapping.f(), mapping.kept_ids(),
                            Index(mask.size()));
}

/// @ingroup core_algorithms
/// @brief Constructs a new @ref tf::mapping_val<Index> from a boolean mask.
///
/// This is a convenience overload that returns a fully constructed
/// bidirectional mapping from the mask. Internally calls the above overload.
///
/// @tparam Index The integer type used for indexing.
/// @tparam Range A range with boolean values indicating which entries to keep.
/// @param mask The boolean mask indicating which entries to retain.
/// @return A populated `tf::mapping_val<Index>`.
///
/// @see tf::mask_to_mapping
template <typename Index, typename Range>
auto mask_to_index_map(const Range &mask) {
  tf::index_map_buffer<Index> mapping;
  tf::mask_to_index_map(mask, mapping);
  return mapping;
}

} // namespace tf
