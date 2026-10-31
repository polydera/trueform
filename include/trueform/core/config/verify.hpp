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
#include <cstdio>
#include <cstdlib>

#ifndef TF_VERIFY
#ifdef NDEBUG
#define TF_VERIFY 0
#else
#define TF_VERIFY 1
#endif
#endif

namespace tf::core {

/// Whether this build checks the library's own contracts. A whole-pass
/// check lives in `if constexpr (tf::core::verification)`, so it compiles in
/// every build and costs nothing where it is off.
inline constexpr bool verification = TF_VERIFY != 0;

[[noreturn]] inline auto verification_failure(const char *contract) -> void {
  std::fprintf(stderr, "trueform: contract broken: %s\n", contract);
  std::abort();
}

inline auto verify(bool holds, const char *contract) -> void {
  if (!holds)
    verification_failure(contract);
}

} // namespace tf::core
