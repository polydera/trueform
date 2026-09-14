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

#include "trueform/cpp/topology/domain_labels.hpp"

#include "trueform/cpp/core/mesh.hpp"
#include "trueform/topology/make_domain_labels.hpp"
#include "trueform/topology/policy/face_membership.hpp"
#include "trueform/topology/policy/manifold_edge_link.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tf::cpp {
namespace detail {

inline auto require_domain_config(tf::domain_config config) -> void {
  constexpr auto known =
      static_cast<int>(tf::domain_config::exclude_outer_shell) |
      static_cast<int>(tf::domain_config::ignore_open_fragments);
  if ((static_cast<int>(config) & ~known) != 0)
    throw std::invalid_argument("make_domain_labels: invalid domain config");
}

} // namespace detail

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int>>
auto make_domain_labels(const mesh<Index, Real, Dims, Ngon> &value,
                        tf::domain_config config)
    -> domain_labels_result<Index> {
  detail::require_domain_config(config);
  if (value.number_of_faces() == 0)
    return {};
  auto labels = tf::make_domain_labels(value.polygons() |
                                           tf::tag(value.face_membership()) |
                                           tf::tag(value.manifold_edge_link()),
                                       config);
  const auto number_of_faces = static_cast<int>(labels.labels.size());
  return domain_labels_result<Index>(
      nd_array<Index>::from_buffer(std::move(labels.labels.data_buffer()),
                                   {number_of_faces, 2}),
      static_cast<Index>(labels.n_domains),
      static_cast<Index>(labels.outer_shell_label));
}

} // namespace tf::cpp
