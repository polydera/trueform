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
template <typename T0, typename T1, typename Hash = std::hash<T0>,
          typename Equal = std::equal_to<T0>,
          typename Alloc = tf::allocator<std::pair<T0, T1>>>
using hash_map = ska2::flat_hash_map<T0, T1, Hash, Equal, Alloc>;
}
