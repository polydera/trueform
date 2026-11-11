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
/// @brief Tag type to request the report alongside a result.
///
/// Pass @ref tf::return_report to an operation that otherwise answers with
/// its status alone to receive the report in its place: the status, the
/// counts that name a refusal's reason, and the diagnostics only the report
/// measures.
struct return_report_t {};

/// @ingroup core
/// @brief Tag instance for requesting the report.
inline constexpr return_report_t return_report{};

} // namespace tf
