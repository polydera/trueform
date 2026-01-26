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
#include "./hash_map.hpp"

namespace tf {
template <typename T, typename Index, typename Hash = std::hash<T>>
class index_hash_map {
public:
  auto f() const -> const tf::hash_map<T, Index, Hash> & { return _f; }

  auto f() -> tf::hash_map<T, Index, Hash> & { return _f; }

  auto kept_ids() -> tf::buffer<T> & { return _kept_ids; }

  auto kept_ids() const -> const tf::buffer<T> & { return _kept_ids; }

  auto clear() {
    _f.clear();
    _kept_ids.clear();
  }

private:
  tf::hash_map<T, Index, Hash> _f;
  tf::buffer<T> _kept_ids;
};
} // namespace tf
