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
#include "./external/hash_map.hpp"
#include "./memory.hpp"
namespace tf {
template <typename T, typename Hash = std::hash<T>,
          typename Equal = std::equal_to<T>,
          typename Alloc = tf::allocator<T>>
using hash_set = ska2::flat_hash_set<T, Hash, Equal, Alloc>;
}
