/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./config/math.hpp"
#include <cmath>

namespace tf {

namespace core {

inline float fast_sqrt(float x) {
#if defined(TF_MATH_HAS_X86_SQRT_INTRINSICS)
  return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
#elif defined(TF_MATH_HAS_ARM_NEON_SQRT_INTRINSICS)
  return vget_lane_f32(vsqrt_f32(vdup_n_f32(x)), 0);
#else
  return std::sqrt(x);
#endif
}

inline double fast_sqrt(double x) {
#if defined(TF_MATH_HAS_X86_SQRT_INTRINSICS)
  return _mm_cvtsd_f64(_mm_sqrt_sd(_mm_set_sd(x), _mm_set_sd(x)));
#elif defined(TF_MATH_HAS_ARM_NEON_SQRT_INTRINSICS)
  return vget_lane_f64(vsqrt_f64(vdup_n_f64(x)), 0);
#else
  return std::sqrt(x);
#endif
}

} // namespace core

template <typename T> auto sqrt(T x) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    return core::fast_sqrt(x);
  } else {
    return std::sqrt(x);
  }
}
} // namespace tf
