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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/csg/detail/supported_outer_shell.hpp"
#include "trueform/cpp/csg/outer_shell.hpp"
#include "trueform/intersect/intersect_config.hpp"
#include "trueform/intersect/intersect_mode.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Ngon,
          std::enable_if_t<cpp::detail::is_supported_outer_shell_v<Index, Real>,
                           int> = 0>
auto outer_shell(Resolver &&resolver, const mesh<Index, Real, 3, Ngon> &value,
                 tf::intersect_config intersect_config =
                     {tf::intersect_mode::primitives |
                      tf::intersect_mode::resolve_contours})
    -> resolver_result_t<Resolver, tf::polygons_buffer<Index, Real, 3, Ngon>> {
  return submit<tf::polygons_buffer<Index, Real, 3, Ngon>>(
      std::forward<Resolver>(resolver), [value, intersect_config] {
        return cpp::outer_shell(value, intersect_config);
      });
}

template <typename Index, typename Real, std::size_t Ngon,
          std::enable_if_t<cpp::detail::is_supported_outer_shell_v<Index, Real>,
                           int> = 0>
auto outer_shell(const mesh<Index, Real, 3, Ngon> &value,
                 tf::intersect_config intersect_config =
                     {tf::intersect_mode::primitives |
                      tf::intersect_mode::resolve_contours})
    -> std::future<tf::polygons_buffer<Index, Real, 3, Ngon>> {
  return async::outer_shell(future_resolver{}, value, intersect_config);
}

} // namespace tf::cpp::async
