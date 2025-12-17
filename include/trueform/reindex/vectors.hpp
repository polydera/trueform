/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/algorithm/parallel_copy.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/index_map.hpp"
#include "../core/vectors.hpp"
#include "../core/vectors_buffer.hpp"
#include "../core/views/indirect_range.hpp"
namespace tf {
template <typename Policy, typename Range0, typename Range1, typename Policy1>
auto reindexed(const tf::vectors<Policy> &vectors,
               const tf::index_map<Range0, Range1> &im,
               tf::vectors<Policy1> &out) {
  tf::parallel_copy(tf::make_indirect_range(im.kept_ids(), vectors), out);
}

template <typename Policy, typename Range0, typename Range1, typename RealT,
          std::size_t Dims>
auto reindexed(const tf::vectors<Policy> &vectors,
               const tf::index_map<Range0, Range1> &im,
               tf::vectors_buffer<RealT, Dims> &out) {
  out.allocate(im.kept_ids().size());
  auto out_p = out.vectors();
  reindexed(vectors, im, out_p);
}

template <typename Policy, typename Range0, typename Range1>
auto reindexed(const tf::vectors<Policy> &vectors,
               const tf::index_map<Range0, Range1> &im) {
  tf::vectors_buffer<tf::coordinate_type<Policy>, tf::coordinate_dims_v<Policy>>
      out;
  reindexed(vectors, im, out);
  return out;
}
} // namespace tf

