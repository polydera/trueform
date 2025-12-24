/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
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
