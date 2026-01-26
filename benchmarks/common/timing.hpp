/**
 * Benchmark timing utilities
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#pragma once

#include <trueform/trueform.hpp>
#include <limits>
#include <functional>

namespace benchmark {

/**
 * Prevent compiler from optimizing away a value.
 *
 * Forces the compiler to treat the value as if it has observable side effects,
 * preventing dead code elimination of benchmark code.
 *
 * @param value Reference to value that should not be optimized away
 */
template <typename T>
inline void do_not_optimize(T const& value) {
#if defined(__clang__) || defined(__GNUC__)
    asm volatile("" : : "r,m"(value) : "memory");
#else
    // Fallback for other compilers
    volatile auto* ptr = &value;
    (void)ptr;
#endif
}

/**
 * Run a benchmark multiple times and return minimum time.
 *
 * @param prepare Preparation function called before each timed run (not timed)
 * @param f Function to benchmark (timed)
 * @param n_iters Number of iterations (default 10)
 * @return Minimum time in milliseconds across all iterations
 */
template <typename PrepareFunc, typename Func>
double min_time_of(PrepareFunc&& prepare, Func&& f, int n_iters = 10) {
    float min_time = std::numeric_limits<float>::max();

    for (int i = 0; i < n_iters; ++i) {
        prepare();

        tf::tick();
        f();
        auto time_ms = tf::tock();

        min_time = std::min(min_time, time_ms);
    }

    return min_time;
}

/**
 * Run a benchmark multiple times and return minimum time (no preparation).
 *
 * @param f Function to benchmark (timed)
 * @param n_iters Number of iterations (default 10)
 * @return Minimum time in milliseconds across all iterations
 */
template <typename Func>
double min_time_of(Func&& f, int n_iters = 10) {
    return min_time_of([](){}, std::forward<Func>(f), n_iters);
}

/**
 * Run a benchmark multiple times and return mean time.
 *
 * @param prepare Preparation function called before each timed run (not timed)
 * @param f Function to benchmark (timed)
 * @param n_iters Number of iterations (default 10)
 * @return Mean time in milliseconds across all iterations
 */
template <typename PrepareFunc, typename Func>
double mean_time_of(PrepareFunc&& prepare, Func&& f, int n_iters = 10) {
    double total_time = 0.0;

    for (int i = 0; i < n_iters; ++i) {
        prepare();

        tf::tick();
        f();
        auto time_ms = tf::tock();

        total_time += time_ms;
    }

    return total_time / n_iters;
}

/**
 * Run a benchmark multiple times and return mean time (no preparation).
 *
 * @param f Function to benchmark (timed)
 * @param n_iters Number of iterations (default 10)
 * @return Mean time in milliseconds across all iterations
 */
template <typename Func>
double mean_time_of(Func&& f, int n_iters = 10) {
    return mean_time_of([](){}, std::forward<Func>(f), n_iters);
}

}  // namespace benchmark
