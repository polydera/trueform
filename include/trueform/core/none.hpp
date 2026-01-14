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

namespace tf {

/// @ingroup core
/// @brief Tag type indicating absence of a value.
///
/// Used as a default template parameter to trigger type deduction
/// or indicate "no value provided".
struct none_t {};

/// @ingroup core
/// @brief Tag instance for absence of value.
inline constexpr none_t none;

} // namespace tf
