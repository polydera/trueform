/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <utility>
namespace tf::linalg {
template <std::size_t Dims, typename Policy> struct rotation : Policy {
  rotation() = default;
  rotation(const Policy &policy) : Policy{policy} {}
  rotation(Policy &&policy) : Policy{std::move(policy)} {}
  using Policy::Policy;
  using Policy::operator=;
  using Policy::operator();
  constexpr static std::size_t n_rows = Dims;
  constexpr static std::size_t n_columns = Dims;
};
} // namespace tf::linalg
