/*
 * Copyright (c) 2025 XLAB
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
#include "typed_reindex_instantiations.hpp"

#include <cstdint>

namespace tf::cpp {
TF_CPP_INSTANTIATE_TYPED_REINDEX(double, std::int64_t, 2);

#define TF_CPP_INSTANTIATE_REINDEX_AT_ARITY(Ngon)                              \
  TF_CPP_INSTANTIATE_MESH_REINDEX(double, std::int64_t, 2, Ngon)

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_REINDEX_AT_ARITY)

/// A concatenation reads two owners, so its second operand crosses the whole
/// matrix while its first is this shard's.
#define TF_CPP_INSTANTIATE_REINDEX_PAIR(Index1, Real1, Ngon0, Ngon1)           \
  TF_CPP_INSTANTIATE_HETEROGENEOUS_PAIR_AT(double, std::int64_t, Real1,        \
                                           Index1, 2, Ngon0, Ngon1)

#define TF_CPP_INSTANTIATE_REINDEX_EDGE_PAIR(Index1, Real1)                    \
  TF_CPP_INSTANTIATE_HETEROGENEOUS_EDGE_PAIR(double, std::int64_t, Real1,      \
                                             Index1, 2)

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON_PAIR(TF_CPP_INSTANTIATE_REINDEX_PAIR)
TF_CPP_MATRIX_FOR_EACH_INDEX_REAL(TF_CPP_INSTANTIATE_REINDEX_EDGE_PAIR)

/// A soup states its own layout: a fixed-block soup is triangles, which the
/// layer always carries, and an offset-block soup is mixed.
TF_CPP_INSTANTIATE_TRIANGLE_SOUP_REINDEX(double, std::int64_t, 2);
#if TF_CPP_MATRIX_HAS_DYNAMIC
TF_CPP_INSTANTIATE_MIXED_SOUP_REINDEX(double, std::int64_t, 2);
#endif

#undef TF_CPP_INSTANTIATE_REINDEX_EDGE_PAIR
#undef TF_CPP_INSTANTIATE_REINDEX_PAIR
#undef TF_CPP_INSTANTIATE_REINDEX_AT_ARITY
} // namespace tf::cpp
