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
/// @brief Tag type for checked parallel execution.
///
/// Pass to parallel algorithms to enable additional verification
/// or synchronization guarantees.
struct checked_t {};

/// @ingroup core
/// @brief Tag instance for checked execution.
static constexpr checked_t checked;

} // namespace tf
