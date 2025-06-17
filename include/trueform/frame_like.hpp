/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include <cstddef>
namespace tf {
template <std::size_t Dims, typename Policy> struct frame_like {

  auto transformation() const -> const auto & {
    return static_cast<const Policy &>(*this).transformation();
  }

  auto inverse_transformation() const -> const auto & {
    return static_cast<const Policy &>(*this).inverse_transformation();
  }
};
} // namespace tf
