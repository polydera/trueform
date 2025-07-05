/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./external/hash_map.hpp"
namespace tf {
template <typename T0, typename T1, typename... Ts>
using hash_map = ska2::flat_hash_map<T0, T1, Ts...>;
}
