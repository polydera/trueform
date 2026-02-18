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

#include "../core/id_handle.hpp"

namespace tf {

/// @ingroup topology_types
/// @brief Type-safe handle for half-edge identifiers.
/// @tparam T The underlying integral type.
template <typename T> struct half_edge_handle : tf::id_handle<T> {
  using tf::id_handle<T>::id_handle;
  static auto invalid() -> half_edge_handle {
    return half_edge_handle{tf::id_handle<T>::_no_id};
  }
};

/// @ingroup topology_types
/// @brief Type-safe handle for edge identifiers.
/// @tparam T The underlying integral type.
template <typename T> struct edge_handle : tf::id_handle<T> {
  using tf::id_handle<T>::id_handle;
  static auto invalid() -> edge_handle {
    return edge_handle{tf::id_handle<T>::_no_id};
  }
};

/// @ingroup topology_types
/// @brief Type-safe handle for vertex identifiers.
/// @tparam T The underlying integral type.
template <typename T> struct vertex_handle : tf::id_handle<T> {
  using tf::id_handle<T>::id_handle;
  static auto invalid() -> vertex_handle {
    return vertex_handle{tf::id_handle<T>::_no_id};
  }
};

/// @ingroup topology_types
/// @brief Type-safe handle for face identifiers.
/// @tparam T The underlying integral type.
template <typename T> struct face_handle : tf::id_handle<T> {
  using tf::id_handle<T>::id_handle;
  static auto invalid() -> face_handle {
    return face_handle{tf::id_handle<T>::_no_id};
  }
};

} // namespace tf
