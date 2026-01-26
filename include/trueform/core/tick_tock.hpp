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

#include <chrono>
#include <iostream>
#include <string_view>

namespace tf {

namespace core {

// Global inline tick start point
inline thread_local std::chrono::steady_clock::time_point tick_start;
} // namespace core

/// @ingroup core_algorithms
/// @brief Start a timing measurement.
///
/// Captures the current time using `std::chrono::steady_clock`. Typically used
/// in conjunction with @ref tock().
inline auto tick() -> void {
  core::tick_start = std::chrono::steady_clock::now();
}

/// @ingroup core_algorithms
/// @brief End a timing measurement and return the elapsed time in milliseconds.
///
/// Measures the duration since the last call to @ref tick().
///
/// @return Elapsed time in milliseconds as a `float`.
inline auto tock() -> float {
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<float, std::milli>(end - core::tick_start)
      .count();
}

/// @ingroup core_algorithms
/// @brief End a timing measurement and print the result with a custom message.
///
/// Measures the duration since the last call to @ref tick() and prints it to
/// `std::cout` along with a user-specified message.
///
/// @param msg A label to display before the timing result.
inline auto tock(std::string_view msg) -> void {
  auto time = tock();
  std::cout << msg << " " << time << " ms" << std::endl;
}

} // namespace tf
