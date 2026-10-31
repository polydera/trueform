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

/** @defgroup linalg Linear Algebra Module
 *  Sparse symmetric positive definite solves: a pattern analyzed once, a
 *  factor per value set, many right-hand sides per factor.
 */

#include "./linalg/analyze_cholesky.hpp" // IWYU pragma: export
#include "./linalg/cholesky_factor.hpp"  // IWYU pragma: export
#include "./linalg/cholesky_pattern.hpp" // IWYU pragma: export
#include "./linalg/cholesky_scratch.hpp" // IWYU pragma: export
#include "./linalg/factor_cholesky.hpp"  // IWYU pragma: export
#include "./linalg/solve_cholesky.hpp"   // IWYU pragma: export
#include "./linalg/symmetric_lower.hpp"  // IWYU pragma: export
