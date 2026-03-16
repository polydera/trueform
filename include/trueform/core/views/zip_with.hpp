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

#include "../apply.hpp"
#include "./mapped_range.hpp"
#include "./zip.hpp"

namespace tf {
namespace views {

/// @ingroup core_ranges
/// @brief Deref functor for @ref tf::zip_with that constructs a typed struct
/// from zipped tuple elements via @ref tf::apply.
///
/// Default-constructible (no lambda), so iterators that store it do not
/// need `std::optional`.
///
/// @tparam Tag The type to construct from the zipped elements.
template <typename Tag> struct zipped_with_dref {
  template <typename T>
  auto operator()(T &&t) const -> decltype(auto) {
    return tf::apply<Tag>(static_cast<T &&>(t));
  }
};

} // namespace views

/// @ingroup core_ranges
/// @brief Zip multiple ranges and map each element tuple into a typed struct.
///
/// Combines @ref tf::zip and @ref tf::make_mapped_range with a deref that
/// constructs `Tag` from the zipped elements via @ref tf::apply.
///
/// @code
/// struct vertex_with_normal {
///     point_view<float, 3> position;
///     unit_vector_view<float, 3> normal;
/// };
///
/// auto view = tf::zip_with<vertex_with_normal>(points, normals);
/// // view[i].position, view[i].normal
/// @endcode
///
/// @tparam Tag The type to construct per element.
/// @tparam Ranges The input range types.
/// @param ranges The ranges to zip.
/// @return A mapped range of `Tag` instances.
template <typename Tag, typename... Ranges>
auto zip_with(Ranges &&...ranges) {
  return tf::make_mapped_range(tf::zip(static_cast<Ranges &&>(ranges)...),
                               views::zipped_with_dref<Tag>{});
}

} // namespace tf
