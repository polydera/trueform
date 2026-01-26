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
#include "./buffer.hpp"

namespace tf {

template <typename Range0, typename Range1>
class index_map {
public:
  /// @brief Default constructor.
  index_map() = default;

  /// @brief Constructs the mapping from two ranges.
  /// @param _f The forward mapping range (e.g., new index for each original ID).
  /// @param _inv_f The list of original IDs that were retained.
  index_map(const Range0 &_f, const Range1 &_inv_f) : _f{_f}, _inv_f{_inv_f} {}

  /// @brief Move constructor from two ranges.
  /// @param _f The forward mapping range.
  /// @param _inv_f The list of retained IDs.
  index_map(Range0 &&_f, Range1 &&_inv_f)
      : _f{std::move(_f)}, _inv_f{std::move(_inv_f)} {}

  /// @brief Returns a mutable reference to the mapping function.
  auto f() -> Range0 & { return _f; }

  /// @brief Returns a const reference to the mapping function.
  auto f() const -> const Range0 & { return _f; }

  /// @brief Returns a mutable reference to the list of kept IDs.
  auto kept_ids() -> Range1 & { return _inv_f; }

  /// @brief Returns a const reference to the list of kept IDs.
  auto kept_ids() const -> const Range1 & { return _inv_f; }

private:
  Range0 _f;
  Range1 _inv_f;
};

template <typename Range0, typename Range1>
auto make_index_map(Range0 &&_f, Range1 &&_kept_ids) {
  return index_map<std::decay_t<Range0>, std::decay_t<Range1>>{
      static_cast<Range0 &&>(_f), static_cast<Range1 &&>(_kept_ids)};
}

template <typename Index>
class index_map_buffer : public index_map<tf::buffer<Index>, tf::buffer<Index>> {};

} // namespace tf
