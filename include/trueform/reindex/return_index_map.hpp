/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

namespace tf {

/// @ingroup reindex
/// @brief Tag type to request index map return.
///
/// Pass @ref tf::return_index_map as an argument to filtering functions
/// to receive the generated @ref tf::index_map_buffer alongside results.
struct return_index_map_t {};

/// @ingroup reindex
/// @brief Tag instance for requesting index map return.
inline constexpr return_index_map_t return_index_map{};

} // namespace tf
