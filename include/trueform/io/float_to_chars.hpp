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
#include <charconv>
#include <cstdio>

// Detect floating-point to_chars support:
// - GCC 11.0+ (libstdc++)
// - Clang 14+ with libc++ (but Apple clang may lag)
// - MSVC 16.0+ (VS 2019)
#if defined(_MSC_VER) && _MSC_VER >= 1920
  // MSVC 2019 (16.0) and later
  #define TF_HAVE_FLOAT_TO_CHARS 1
#elif defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11
  // GCC 11.0+
  #define TF_HAVE_FLOAT_TO_CHARS 1
#elif defined(__clang__) && !defined(__apple_build_version__) && __clang_major__ >= 14
  // Clang 14+ (non-Apple)
  #define TF_HAVE_FLOAT_TO_CHARS 1
#else
  // Apple clang, older compilers - use fallback
  #define TF_HAVE_FLOAT_TO_CHARS 0
#endif

namespace tf::io {

inline auto float_to_chars(char *first, char *last, float value) -> char * {
#if TF_HAVE_FLOAT_TO_CHARS
  auto result = std::to_chars(first, last, value, std::chars_format::general);
  return result.ptr;
#else
  // Portable fallback using snprintf
  // Print with enough precision to round-trip (FLT_DECIMAL_DIG = 9)
  int written = std::snprintf(first, last - first, "%.9g", value);

  if (written < 0 || first + written >= last) {
    return last; // Error indicator
  }

  return first + written;
#endif
}

} // namespace tf::io
