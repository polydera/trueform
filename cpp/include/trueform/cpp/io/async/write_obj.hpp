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
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/io/write_obj.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon, std::enable_if_t<Dims == 3, int> = 0>
auto write_obj(Resolver &&resolver,
               const cpp::mesh<Index, Real, Dims, Ngon> &value)
    -> resolver_result_t<Resolver, nd_array<std::int8_t>> {
  return submit<nd_array<std::int8_t>>(
      std::forward<Resolver>(resolver),
      [value] { return cpp::write_obj(value); });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto write_obj(const cpp::mesh<Index, Real, Dims, Ngon> &value)
    -> std::future<nd_array<std::int8_t>> {
  return async::write_obj(future_resolver{}, value);
}

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon, std::enable_if_t<Dims == 3, int> = 0>
auto write_obj(Resolver &&resolver,
               const cpp::mesh<Index, Real, Dims, Ngon> &value,
               const std::filesystem::path &path)
    -> resolver_result_t<Resolver, bool> {
  auto owned_path = path;
  return submit<bool>(std::forward<Resolver>(resolver),
                      [value, path = std::move(owned_path)] {
                        return cpp::write_obj(value, path);
                      });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto write_obj(const cpp::mesh<Index, Real, Dims, Ngon> &value,
               const std::filesystem::path &path) -> std::future<bool> {
  return async::write_obj(future_resolver{}, value, path);
}

} // namespace tf::cpp::async
