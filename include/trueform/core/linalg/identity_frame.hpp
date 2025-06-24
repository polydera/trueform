/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../transformation_like.hpp"
namespace tf::linalg {

template <typename T, std::size_t Dims> class identity_frame {
public:
  using coordinate_type = std::decay_t<T>;

  auto transformation() const -> identity_transformation<T, Dims> { return {}; }

  auto inverse_transformation() const -> identity_transformation<T, Dims> {
    return {};
  }
};

} // namespace tf::linalg
