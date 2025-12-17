/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./external/hash_map.hpp"
namespace tf {
template <typename T0, typename T1, typename... Ts>
using hash_map = ska2::flat_hash_map<T0, T1, Ts...>;
}
