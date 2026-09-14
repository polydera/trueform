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
#include "trueform/cpp/core/detail/minted_mesh_result.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/io/bytes.hpp"
#include "trueform/cpp/io/read_stl.hpp"

#include <cstddef>
#include <cstdint>
#include <future>
#include <string>
#include <string_view>
#include <utility>

namespace tf::cpp::async {

template <typename Index = default_index_t, typename Resolver>
auto read_stl(Resolver &&resolver, std::string_view path)
    -> resolver_result_t<Resolver,
                         cpp::detail::minted_mesh_result_t<Index, float, 3>> {
  auto owned_path = std::string(path);
  return submit<cpp::detail::minted_mesh_result_t<Index, float, 3>>(
      std::forward<Resolver>(resolver),
      [path = std::move(owned_path)] { return cpp::read_stl<Index>(path); });
}

template <typename Index = default_index_t>
auto read_stl(std::string_view path)
    -> std::future<cpp::detail::minted_mesh_result_t<Index, float, 3>> {
  return async::read_stl<Index>(future_resolver{}, path);
}

template <typename Index = default_index_t, typename Resolver>
auto read_stl(Resolver &&resolver, io_bytes bytes)
    -> resolver_result_t<Resolver,
                         cpp::detail::minted_mesh_result_t<Index, float, 3>> {
  return submit<cpp::detail::minted_mesh_result_t<Index, float, 3>>(
      std::forward<Resolver>(resolver), [bytes = std::move(bytes)]() mutable {
        return cpp::read_stl<Index>(std::move(bytes));
      });
}

template <typename Index = default_index_t>
auto read_stl(io_bytes bytes)
    -> std::future<cpp::detail::minted_mesh_result_t<Index, float, 3>> {
  return async::read_stl<Index>(future_resolver{}, std::move(bytes));
}

template <typename Index = default_index_t, typename Resolver>
auto read_stl(Resolver &&resolver, const std::int8_t *data, std::size_t size)
    -> resolver_result_t<Resolver,
                         cpp::detail::minted_mesh_result_t<Index, float, 3>> {
  return async::read_stl<Index>(std::forward<Resolver>(resolver),
                                io_bytes::copy(data, size));
}

template <typename Index = default_index_t>
auto read_stl(const std::int8_t *data, std::size_t size)
    -> std::future<cpp::detail::minted_mesh_result_t<Index, float, 3>> {
  return async::read_stl<Index>(future_resolver{}, data, size);
}

} // namespace tf::cpp::async
