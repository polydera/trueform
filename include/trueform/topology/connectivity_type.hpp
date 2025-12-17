/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

namespace tf {

/// @brief Connectivity types for mesh traversal algorithms.
///
/// Determines how components are connected:
/// - manifold_edge: only through manifold edges (separates at boundaries/non-manifold)
/// - edge: through any shared edge
/// - vertex: through any shared vertex (most permissive)
enum class connectivity_type { manifold_edge, edge, vertex };

} // namespace tf
