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
 * Author: Ziga Sajovic
 */

#include "trueform/python/core/primitive_wrapper.hpp"

namespace tf::py {

auto register_primitive_wrappers(nanobind::module_ &m) -> void {
  register_primitive_wrapper<2, float>(m, "PrimitiveWrapperFloat2D");
  register_primitive_wrapper<3, float>(m, "PrimitiveWrapperFloat3D");
  register_primitive_wrapper<2, double>(m, "PrimitiveWrapperDouble2D");
  register_primitive_wrapper<3, double>(m, "PrimitiveWrapperDouble3D");
}

// Explicit template instantiations
template class primitive_wrapper<2, float>;
template class primitive_wrapper<3, float>;
template class primitive_wrapper<2, double>;
template class primitive_wrapper<3, double>;

} // namespace tf::py
