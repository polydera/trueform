/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/frame_like.hpp"
#include "../core/transformation_like.hpp"
#include "./tree.hpp"

namespace tf::spatial {

template <typename Index, typename RealT, std::size_t Dims, typename Policy>
class model : public Policy {
public:
  using real_t = RealT;
  using index_t = Index;
  using transformation_t = identity_transformation<real_t, Dims>;
  using frame_t = identity_frame<real_t, Dims>;

  model(const tf::tree<Index, RealT, Dims> &_tree, const Policy &policy)
      : Policy{policy}, _tree{_tree} {}

  auto tree() const -> const tf::tree<Index, RealT, Dims> & { return _tree; }

  auto transformation() const -> tf::identity_transformation<RealT, Dims> {
    return identity_transformation<RealT, Dims>{};
  }

  auto inverse_transformation() const
      -> tf::identity_transformation<RealT, Dims> {
    return identity_transformation<RealT, Dims>{};
  }

  auto frame() const -> tf::identity_frame<RealT, Dims> {
    return identity_frame<RealT, Dims>{};
  }

  auto aabb() const { return _tree.aabb(); }

private:
  const tf::tree<Index, RealT, Dims> &_tree;
};

template <typename Index, typename RealT, std::size_t Dims, typename Policy>
auto make_model(const tf::tree<Index, RealT, Dims> &tree, Policy &&policy) {
  return model<Index, RealT, Dims, std::decay_t<Policy>>{tree, policy};
}
} // namespace tf::spatial
