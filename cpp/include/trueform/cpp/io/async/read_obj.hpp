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

#include "trueform/core/static_size.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/detail/minted_mesh_result.hpp"
#include "trueform/cpp/io/bytes.hpp"
#include "trueform/cpp/io/read_obj.hpp"

#include <cstddef>
#include <cstdint>
#include <future>
#include <string>
#include <string_view>
#include <utility>

namespace tf::cpp::async {

template <typename Index, typename Real, std::size_t Ngon = tf::dynamic_size,
          typename Resolver>
auto read_obj(Resolver &&resolver, std::string_view path)
    -> resolver_result_t<Resolver,
                         cpp::detail::minted_mesh_result_t<Index, Real, Ngon>> {
  auto owned_path = std::string(path);
  return submit<cpp::detail::minted_mesh_result_t<Index, Real, Ngon>>(
      std::forward<Resolver>(resolver), [path = std::move(owned_path)] {
        return cpp::read_obj<Index, Real, Ngon>(path);
      });
}

template <typename Index, typename Real, std::size_t Ngon = tf::dynamic_size>
auto read_obj(std::string_view path)
    -> std::future<cpp::detail::minted_mesh_result_t<Index, Real, Ngon>> {
  return async::read_obj<Index, Real, Ngon>(future_resolver{}, path);
}

template <typename Index, typename Real, std::size_t Ngon = tf::dynamic_size,
          typename Resolver>
auto read_obj(Resolver &&resolver, io_bytes bytes)
    -> resolver_result_t<Resolver,
                         cpp::detail::minted_mesh_result_t<Index, Real, Ngon>> {
  return submit<cpp::detail::minted_mesh_result_t<Index, Real, Ngon>>(
      std::forward<Resolver>(resolver), [bytes = std::move(bytes)]() mutable {
        return cpp::read_obj<Index, Real, Ngon>(std::move(bytes));
      });
}

template <typename Index, typename Real, std::size_t Ngon = tf::dynamic_size>
auto read_obj(io_bytes bytes)
    -> std::future<cpp::detail::minted_mesh_result_t<Index, Real, Ngon>> {
  return async::read_obj<Index, Real, Ngon>(future_resolver{},
                                            std::move(bytes));
}

template <typename Index, typename Real, std::size_t Ngon = tf::dynamic_size,
          typename Resolver>
auto read_obj(Resolver &&resolver, const std::int8_t *data, std::size_t size)
    -> resolver_result_t<Resolver,
                         cpp::detail::minted_mesh_result_t<Index, Real, Ngon>> {
  return async::read_obj<Index, Real, Ngon>(std::forward<Resolver>(resolver),
                                            io_bytes::copy(data, size));
}

template <typename Index, typename Real, std::size_t Ngon = tf::dynamic_size>
auto read_obj(const std::int8_t *data, std::size_t size)
    -> std::future<cpp::detail::minted_mesh_result_t<Index, Real, Ngon>> {
  return async::read_obj<Index, Real, Ngon>(future_resolver{}, data, size);
}

} // namespace tf::cpp::async
