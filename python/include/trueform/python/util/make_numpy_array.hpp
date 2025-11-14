/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#pragma once

#include <initializer_list>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/python/util/make_capsule.hpp>

namespace tf::py {

/**
 * Create a numpy array from raw pointer with proper ownership transfer
 * Handles empty arrays safely by using a shared dummy allocation
 * Shape is explicit, type T is inferred from the pointer
 */
template <typename Shape, typename T>
auto make_numpy_array(T *data, std::initializer_list<size_t> shape) {
  auto capsule = make_capsule<T>(data);
  return nanobind::ndarray<nanobind::numpy, T, Shape>(
      data ? data : reinterpret_cast<T*>(capsule.data()), shape, capsule);
}

/**
 * Create a numpy array from raw pointer with proper ownership transfer
 * Handles empty arrays safely by using a shared dummy allocation
 * Type T is inferred from the pointer, no shape constraint
 * Delegates to the shape-explicit version with shape<-1>
 */
template <typename T>
auto make_numpy_array(T *data, std::initializer_list<size_t> shape) {
  return make_numpy_array<nanobind::shape<-1>>(data, shape);
}

} // namespace tf::py
