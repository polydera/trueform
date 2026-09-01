/*
 * Copyright (c) 2026 XLAB
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
/// @brief Tag type to request the refused ids alongside a result.
///
/// Pass @ref tf::return_refused to an operation whose per-carrier
/// answer may be empty to additionally receive, ascending, the ids of
/// the carriers whose triangulation was refused. Emptiness is always
/// the answer; the list names whose emptiness it was.
struct return_refused_t {};

/// @ingroup core
/// @brief Tag instance for requesting the refused ids.
inline constexpr return_refused_t return_refused{};

} // namespace tf
