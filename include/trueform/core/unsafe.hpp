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

namespace tf {

/// @ingroup core
/// @brief Tag type to skip validation or normalization.
///
/// Pass to constructors/factories that normally validate or normalize
/// input when you guarantee the input is already valid.
/// Example: `tf::make_unit_vector(tf::unsafe, already_normalized_vec)`
struct unsafe_t {};

/// @ingroup core
/// @brief Tag instance to skip validation.
static constexpr unsafe_t unsafe;

} // namespace tf
