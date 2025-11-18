/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <charconv>
#include <cstdio>
#include <system_error>

// Detect if to_chars
#if defined(__APPLE__)
#define TF_HAVE_FLOAT_TO_CHARS 0
#else
#define TF_HAVE_FLOAT_TO_CHARS 1 // Linux, Windows have full support
#endif

inline std::to_chars_result float_to_chars(char *first, char *last,
                                           float value) {
#if TF_HAVE_FLOAT_TO_CHARS
  // Safe, fast path
  return std::to_chars(first, last, static_cast<double>(value),
                       std::chars_format::general);
#else
  // Portable fallback
  // Print with enough precision to round-trip (FLT_DECIMAL_DIG = 9)
  int written = std::snprintf(first, last - first, "%.9g", value);

  if (written < 0 || first + written >= last) {
    return {last, std::errc::value_too_large};
  }

  return {first + written, std::errc{}};
#endif
}
