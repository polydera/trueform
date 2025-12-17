/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
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
