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
/// @brief Tag type to request a "complete" / fully-featured variant.
///
/// Pass to factories that have both a minimal default and a richer
/// fully-featured variant. Example: `tf::read_obj(path, tf::complete)`
/// reads positions, normals, textures, groups and objects instead of
/// only positions and faces.
struct complete_t {};

/// @ingroup core
/// @brief Tag instance to request the complete variant.
inline constexpr complete_t complete{};

} // namespace tf
