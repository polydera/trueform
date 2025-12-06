/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../none.hpp"

namespace tf {

template <typename U> auto operator|(U &&u, none_t) -> U && {
  return static_cast<U &&>(u);
}

inline auto tag(none_t) -> none_t { return none; }

template <typename U> auto tag(none_t, U &&u) -> U && {
  return static_cast<U &&>(u);
}

} // namespace tf
