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
#include "topology_analysis_impl.hpp"

#include <cstdint>

namespace tf::cpp {

template auto cell_membership(const nd_array<std::int32_t> &, std::int32_t)
    -> offset_blocked_buffer<std::int32_t, std::int32_t>;
template auto
cell_membership(const offset_blocked_buffer<std::int32_t, std::int32_t> &,
                std::int32_t)
    -> offset_blocked_buffer<std::int32_t, std::int32_t>;
template auto vertex_link_edges(const nd_array<std::int32_t> &, std::int32_t)
    -> offset_blocked_buffer<std::int32_t, std::int32_t>;
template auto
vertex_link_faces(const nd_array<std::int32_t> &,
                  const offset_blocked_buffer<std::int32_t, std::int32_t> &)
    -> offset_blocked_buffer<std::int32_t, std::int32_t>;
template auto
vertex_link_faces(const offset_blocked_buffer<std::int32_t, std::int32_t> &,
                  const offset_blocked_buffer<std::int32_t, std::int32_t> &)
    -> offset_blocked_buffer<std::int32_t, std::int32_t>;

template auto
manifold_edge_link(const nd_array<std::int32_t> &,
                   const offset_blocked_buffer<std::int32_t, std::int32_t> &)
    -> nd_array<std::int32_t>;
template auto
manifold_edge_link(const offset_blocked_buffer<std::int32_t, std::int32_t> &,
                   const offset_blocked_buffer<std::int32_t, std::int32_t> &)
    -> offset_blocked_buffer<std::int32_t, std::int32_t>;
template auto
face_link(const nd_array<std::int32_t> &,
          const offset_blocked_buffer<std::int32_t, std::int32_t> &)
    -> offset_blocked_buffer<std::int32_t, std::int32_t>;
template auto
face_link(const offset_blocked_buffer<std::int32_t, std::int32_t> &,
          const offset_blocked_buffer<std::int32_t, std::int32_t> &)
    -> offset_blocked_buffer<std::int32_t, std::int32_t>;

} // namespace tf::cpp
