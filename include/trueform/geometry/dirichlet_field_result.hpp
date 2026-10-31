/*
 * Copyright (c) 2026 XLAB
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
#include "../core/buffer.hpp"
#include "./dirichlet_field_status.hpp"

namespace tf {

/// @ingroup geometry
/// @brief What @ref tf::make_dirichlet_field solved: one value per point, and
/// the @ref tf::dirichlet_field_status its system answered.
///
/// `field` states the fixed values at the fixed points whatever happened. A
/// solved field states its answer everywhere else; a refused one states zero
/// there.
///
/// @tparam RealT The type the field is stated in.
template <typename RealT> struct dirichlet_field_result {
  tf::buffer<RealT> field;
  tf::dirichlet_field_status status;
};

} // namespace tf
