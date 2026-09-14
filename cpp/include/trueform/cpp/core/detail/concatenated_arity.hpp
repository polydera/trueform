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

#include "trueform/core/static_size.hpp"

#include <cstddef>

namespace tf::cpp::detail {

/// @brief The arity two carriers state together.
///
/// Triangles beside triangles are triangles; anything else must state a face's
/// own size, so it is mixed. One rule, for every entry that puts two carriers
/// into one output.
template <std::size_t Ngon0, std::size_t Ngon1>
inline constexpr std::size_t concatenated_arity_v =
    (Ngon0 == 3 && Ngon1 == 3) ? std::size_t{3} : tf::dynamic_size;

} // namespace tf::cpp::detail
