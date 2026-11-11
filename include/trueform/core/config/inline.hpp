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

/// A per-element step the compiler must fold into its one hot caller, and a
/// cold path it must keep out of it: the compiler's own inlining heuristic
/// reads a hot step's size, not its call count.
#if defined(_MSC_VER)
#define TF_FORCE_INLINE __forceinline
#define TF_NOINLINE __declspec(noinline)
#else
#define TF_FORCE_INLINE __attribute__((always_inline)) inline
#define TF_NOINLINE __attribute__((noinline))
#endif
