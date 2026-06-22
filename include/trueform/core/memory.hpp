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

#include <cstddef>
#include <new>
#include <vector>

#ifdef TF_WITH_MIMALLOC
#include <mimalloc.h>
#endif

namespace tf {

/// Typed so alignof(T) rides into a plain void(*)(void*) noexcept deallocate<T>,
/// usable directly as a VTK/nanobind free callback.
template <typename T> auto allocate(std::size_t n) -> T * {
#ifdef TF_WITH_MIMALLOC
  return static_cast<T *>(::mi_malloc_aligned(n * sizeof(T), alignof(T)));
#else
  return static_cast<T *>(
      ::operator new(n * sizeof(T), std::align_val_t{alignof(T)}));
#endif
}

template <typename T> auto deallocate(void *p) noexcept -> void {
#ifdef TF_WITH_MIMALLOC
  ::mi_free(p);
#else
  ::operator delete(p, std::align_val_t{alignof(T)});
#endif
}

/// std::allocator adapter over the seam; stateless.
template <typename T> struct allocator {
  using value_type = T;

  allocator() noexcept = default;
  template <typename U> allocator(const allocator<U> &) noexcept {}

  auto allocate(std::size_t n) -> T * { return tf::allocate<T>(n); }
  auto deallocate(T *p, std::size_t) noexcept -> void { tf::deallocate<T>(p); }

  template <typename U>
  auto operator==(const allocator<U> &) const noexcept -> bool { return true; }
  template <typename U>
  auto operator!=(const allocator<U> &) const noexcept -> bool { return false; }
};

namespace core {
template <typename T> using std_vector = std::vector<T, tf::allocator<T>>;
} // namespace core

} // namespace tf
