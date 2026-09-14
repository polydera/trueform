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
#include "mesh_primitives_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_TYPED_MESH_PRIMITIVES(Real, Index)                  \
  template auto make_sphere_mesh<Index, Real>(Real, std::int32_t,              \
                                              std::int32_t) ->                 \
      typename detail::minted_mesh_result<Index, Real, 3>::type;               \
  template auto make_cylinder_mesh<Index, Real>(Real, Real, std::int32_t) ->   \
      typename detail::minted_mesh_result<Index, Real, 3>::type;               \
  template auto make_box_mesh<Index, Real>(Real, Real, Real)                   \
      ->typename detail::minted_mesh_result<Index, Real, 3>::type;             \
  template auto make_box_mesh<Index, Real>(Real, Real, Real, std::int32_t,     \
                                           std::int32_t, std::int32_t) ->      \
      typename detail::minted_mesh_result<Index, Real, 3>::type;               \
  template auto make_plane_mesh<Index, Real>(Real, Real, std::int32_t,         \
                                             std::int32_t) ->                  \
      typename detail::minted_mesh_result<Index, Real, 3>::type;               \
  template auto make_tube_mesh<Index, Real>(                                   \
      const offset_blocked_buffer<Index, Index> &, const nd_array<Real> &,     \
      Real, std::int32_t) ->                                                   \
      typename detail::minted_mesh_result<Index, Real, 3>::type

TF_CPP_INSTANTIATE_TYPED_MESH_PRIMITIVES(float, std::int64_t);

#undef TF_CPP_INSTANTIATE_TYPED_MESH_PRIMITIVES

} // namespace tf::cpp
