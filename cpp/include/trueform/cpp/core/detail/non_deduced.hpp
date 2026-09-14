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

namespace tf::cpp::detail {

/// A parameter that must not take part in deduction: the type it is read at is
/// the one the carriers beside it already named.
template <typename T> struct non_deduced {
  using type = T;
};
template <typename T> using non_deduced_t = typename non_deduced<T>::type;

} // namespace tf::cpp::detail
