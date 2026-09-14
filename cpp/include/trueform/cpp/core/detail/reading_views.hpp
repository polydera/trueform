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

#include "trueform/core/range.hpp"
#include "trueform/core/static_size.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp::detail {

/// @brief Whether a view reads blocks of `Arity` values of `Scalar`.
///
/// A caller holding its storage const and a caller still writing through it
/// hand the SAME reading in two spellings, so what an assembly asks of a view
/// is what it reads — the scalar and the block arity — and not whether the
/// caller's pointer happens to be const. A view that has no flat storage
/// behind it answers false, which is what keeps the assembly from swallowing
/// every two-argument call.
template <typename View, typename Scalar, std::size_t Arity, typename = void>
struct reads_blocks : std::false_type {};

template <typename View, typename Scalar, std::size_t Arity>
struct reads_blocks<
    View, Scalar, Arity,
    std::enable_if_t<
        std::is_same<
            std::remove_const_t<std::remove_pointer_t<std::decay_t<
                decltype(std::declval<const View &>().begin().base_iter())>>>,
            Scalar>::value &&
        tf::static_size_v<std::decay_t<
            decltype(*std::declval<const View &>().begin())>> == Arity>>
    : std::true_type {};

template <typename View, typename Scalar, std::size_t Arity>
inline constexpr bool reads_blocks_v = reads_blocks<View, Scalar, Arity>::value;

/// @brief The same flat span, read as const.
///
/// A reading is const whatever the caller holds, and this is the one place the
/// pointer narrows.
template <typename Iterator>
auto const_range_of(Iterator first, Iterator last) {
  using element = std::remove_pointer_t<Iterator>;
  return tf::make_range(static_cast<const element *>(first),
                        static_cast<const element *>(last));
}

template <typename Iterator>
auto const_range_of(Iterator first, std::size_t length) {
  using element = std::remove_pointer_t<Iterator>;
  return tf::make_range(static_cast<const element *>(first), length);
}

} // namespace tf::cpp::detail
