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
#include "./compiled_expr.hpp"
#include <cstdint>

namespace tf::csg::detail {

/// @ingroup csg
/// @brief Word/bit decomposition of a single operand id.
inline auto make_word_mask(int operand_id) -> compiled_expr::word_mask {
  return {operand_id / 32,
          std::uint32_t(1) << (static_cast<unsigned>(operand_id) % 32u)};
}

} // namespace tf::csg::detail
