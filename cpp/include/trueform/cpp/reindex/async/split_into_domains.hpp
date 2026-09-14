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
#include "trueform/cpp/reindex/split_into_domains.hpp"
#include "trueform/cpp/topology/domain_labels.hpp"

#include <cstddef>
#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon>
auto split_into_domains(Resolver &&resolver,
                        const mesh<Index, Real, Dims, Ngon> &value,
                        const domain_labels_result<Index> &labels) {
  using result_type = split_domains_result<Index, Real, Dims, Ngon>;
  return submit<result_type>(std::forward<Resolver>(resolver), [value, labels] {
    return cpp::split_into_domains<Index, Real, Dims, Ngon>(value, labels);
  });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto split_into_domains(const mesh<Index, Real, Dims, Ngon> &value,
                        const domain_labels_result<Index> &labels)
    -> std::future<split_domains_result<Index, Real, Dims, Ngon>> {
  return async::split_into_domains<future_resolver, Index, Real, Dims, Ngon>(
      future_resolver{}, value, labels);
}

} // namespace tf::cpp::async
