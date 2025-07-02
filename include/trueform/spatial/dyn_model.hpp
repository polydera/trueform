/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/frame_like.hpp"
#include "./model.hpp"

namespace tf::spatial {

template <std::size_t Dims, typename FPolicy, typename Index, typename RealT,
          typename Policy>
class dyn_model : public model<Index, RealT, Dims, Policy> {
  using base_t = model<Index, RealT, Dims, Policy>;

public:
  dyn_model(tf::frame_like<Dims, FPolicy> _frame,
            const tf::tree<Index, RealT, Dims> &_tree, const Policy &policy)
      : base_t{_tree, policy}, _frame{_frame} {}

  auto transformation() const -> const auto & {
    return _frame.transformation();
  }

  auto inverse_transformation() const -> const auto & {
    return _frame.inverse_transformation();
  }

  auto frame() const -> const auto & { return _frame; }

private:
  tf::frame_like<Dims, FPolicy> _frame;
};
template <std::size_t Dims, typename FPolicy, typename Index, typename RealT,
          typename Policy>
auto make_dyn_model(const frame_like<Dims, FPolicy> &frame,
                    const tf::tree<Index, RealT, Dims> &tree, Policy &&policy) {
  return dyn_model<Dims, FPolicy, Index, RealT, std::decay_t<Policy>>{frame, tree, policy};
}

} // namespace tf::spatial
