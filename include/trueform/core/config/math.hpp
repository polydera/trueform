/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) ||               \
    defined(_M_IX86)
#include <immintrin.h>
#define TF_MATH_HAS_X86_SQRT_INTRINSICS 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define TF_MATH_HAS_ARM_NEON_SQRT_INTRINSICS 1
#endif
