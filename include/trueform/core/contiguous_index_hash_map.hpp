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
#include "./edges.hpp"
#include "./hash_map.hpp"
#include "./index_hash_map.hpp"

namespace tf {
template <typename Range, typename T, typename Index, typename Hash>
auto make_contiguous_index_hash_map(const Range &r,
                                    tf::hash_map<T, Index, Hash> &ihm,
                                    Index current = 0) {
  ihm.f().reserve(r.size() / 2);
  for (const auto &e : r) {
    if (ihm.find(e) == ihm.end()) {
      ihm[e] = current++;
    }
  }
}

template <typename Range, typename T, typename Index, typename Hash>
auto make_contiguous_index_hash_map(const Range &r,
                                    tf::index_hash_map<T, Index, Hash> &ihm,
                                    Index current = 0) {
  ihm.f().reserve(r.size() / 2);
  ihm.kept_ids().reserve(r.size() / 2);
  for (const auto &e : r) {
    if (ihm.f().find(e) == ihm.f().end()) {
      ihm.kept_ids().push_back(e);
      ihm.f()[e] = current++;
    }
  }
}

template <typename Policy, typename T, typename Index, typename Hash>
auto make_contiguous_index_hash_map(const tf::edges<Policy> &edges,
                                    tf::index_hash_map<T, Index, Hash> &ihm,
                                    Index current = 0) {
  ihm.f().reserve(edges.size() / 2);
  ihm.kept_ids().reserve(edges.size() / 2);
  for (const auto &e : edges) {
    if (ihm.f().find(e[0]) == ihm.f().end()) {
      ihm.kept_ids().push_back(e[0]);
      ihm.f()[e[0]] = current++;
    }
    if (ihm.f().find(e[1]) == ihm.f().end()) {
      ihm.kept_ids().push_back(e[1]);
      ihm.f()[e[1]] = current++;
    }
  }
}
} // namespace tf
