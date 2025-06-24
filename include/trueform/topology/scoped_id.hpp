/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
namespace tf {
template <typename Index, typename SubIndex> struct scoped_id {
  Index id;
  SubIndex sub_id;
};

template <typename Index, typename SubIndex>
auto make_scoped_id(const Index &id, const SubIndex &sub_id) {
  return scoped_id<Index, SubIndex>{id, sub_id};
}

} // namespace tf
