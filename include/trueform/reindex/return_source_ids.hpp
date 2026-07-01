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

namespace tf {

/// @ingroup reindex
/// @brief Tag type to request source-id return from a split.
///
/// Pass @ref tf::return_source_ids to a split function to additionally
/// receive, per output piece, the original element ids that compose it
/// (face ids for polygons, segment ids for segments) as a
/// @ref tf::offset_block_buffer whose blocks run parallel to the output
/// pieces. This is a provenance map (piece -> source elements), not an
/// @ref tf::index_map_buffer.
struct return_source_ids_t {};

/// @ingroup reindex
/// @brief Tag instance for requesting source-id return.
inline constexpr return_source_ids_t return_source_ids{};

} // namespace tf
