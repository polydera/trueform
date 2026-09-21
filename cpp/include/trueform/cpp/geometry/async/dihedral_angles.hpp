/*
 * Copyright (c) 2026 XLAB
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

#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/geometry/dihedral_angles.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon, std::enable_if_t<Dims == 3, int> = 0>
auto dihedral_angles(Resolver &&resolver,
                     const cpp::mesh<Index, Real, Dims, Ngon> &value) {
  return submit<dihedral_angles_result<Index, Real>>(
      std::forward<Resolver>(resolver),
      [value] { return cpp::dihedral_angles<Index, Real, Dims, Ngon>(value); });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto dihedral_angles(const cpp::mesh<Index, Real, Dims, Ngon> &value)
    -> std::future<dihedral_angles_result<Index, Real>> {
  return async::dihedral_angles<future_resolver, Index, Real, Dims, Ngon>(
      future_resolver{}, value);
}

} // namespace tf::cpp::async
