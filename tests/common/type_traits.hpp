/**
 * @file type_traits.hpp
 * @brief Type definitions for parametrized testing
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#pragma once

#include <cstdint>
#include <tuple>

namespace tf::test {

/**
 * @brief Type pair for parametrized tests
 * @tparam index_t Index type (int32_t or int64_t)
 * @tparam real_t Real type (float or double)
 */
template <typename index_t, typename real_t>
struct type_pair {
    using index_type = index_t;
    using real_type = real_t;
};

/**
 * @brief All type combinations for comprehensive testing
 *
 * Tests all combinations of:
 * - Index types: int32_t, int64_t
 * - Real types: float, double
 */
using all_type_pairs = std::tuple<
    type_pair<std::int32_t, float>,
    type_pair<std::int32_t, double>,
    type_pair<std::int64_t, float>,
    type_pair<std::int64_t, double>
>;

/**
 * @brief Common type pairs for faster test runs
 *
 * Subset of type combinations for quick validation:
 * - int32_t/float (common case)
 * - int64_t/double (high precision case)
 */
using common_type_pairs = std::tuple<
    type_pair<std::int32_t, float>,
    type_pair<std::int64_t, double>
>;

/**
 * @brief Index types for tests that only vary index type
 */
using index_types = std::tuple<std::int32_t, std::int64_t>;

/**
 * @brief Real types for tests that only vary real type
 */
using real_types = std::tuple<float, double>;

/**
 * @brief Type pair with single dynamic flag for 1-mesh tests
 */
template <typename IndexT, typename RealT, bool Dynamic>
struct type_pair_dyn {
    using index_type = IndexT;
    using real_type = RealT;
    static constexpr bool is_dynamic = Dynamic;
};

/**
 * @brief Type pair with two dynamic flags for 2-mesh tests
 */
template <typename IndexT, typename RealT, bool Dynamic1, bool Dynamic2>
struct type_pair_dyn2 {
    using index_type = IndexT;
    using real_type = RealT;
    static constexpr bool is_dynamic1 = Dynamic1;
    static constexpr bool is_dynamic2 = Dynamic2;
};

} // namespace tf::test
