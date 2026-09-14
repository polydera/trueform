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

#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/topology/domain_labels.hpp"
#include "trueform/topology/domain_config.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon, std::enable_if_t<Dims == 3, int> = 0>
auto make_domain_labels(Resolver &&resolver,
                        const cpp::mesh<Index, Real, Dims, Ngon> &value,
                        tf::domain_config config = tf::domain_config::none)
    -> resolver_result_t<Resolver, domain_labels_result<Index>> {
  return submit<domain_labels_result<Index>>(
      std::forward<Resolver>(resolver),
      [value, config] { return cpp::make_domain_labels(value, config); });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto make_domain_labels(const cpp::mesh<Index, Real, Dims, Ngon> &value,
                        tf::domain_config config = tf::domain_config::none)
    -> std::future<domain_labels_result<Index>> {
  return async::make_domain_labels(future_resolver{}, value, config);
}

} // namespace tf::cpp::async
