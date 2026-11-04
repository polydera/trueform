/**
 * @file arrangement_builders_int32_int32.cpp
 * @brief The compiled one-operand arrangement tier for int32 indices, int32
 * coordinates: an input already on its lattice, as the tet entrance arranges
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "arrangement_builders_impl.hpp"

#include <trueform/exact/int32.hpp>

#include <cstdint>

namespace tf::test {
namespace {
using tri = form_t<std::int32_t, tf::exact::int32, 3>;
} // namespace

template self_arrangement_t<tri>
build_self_arrangement<tri>(const tri &, tf::arrangement_config,
                            tf::arrangement::near_identity_request);

} // namespace tf::test
