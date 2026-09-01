/**
 * @file boolean_builders_int64_double.cpp
 * @brief The compiled pairwise boolean for int64 indices, double coordinates
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "csg_readers_impl.hpp"
#include "tagged_operand.hpp"

#include <cstdint>

namespace tf::test {
namespace {
using tri = form_t<std::int64_t, double, 3>;
using dyn = form_t<std::int64_t, double, tf::dynamic_size>;
} // namespace

template boolean_result_t<tri, tri>
boolean_of<tri, tri>(const tri &, const tri &, tf::boolean_op);

template boolean_result_t<tri, dyn>
boolean_of<tri, dyn>(const tri &, const dyn &, tf::boolean_op);

template boolean_result_t<dyn, tri>
boolean_of<dyn, tri>(const dyn &, const tri &, tf::boolean_op);

template boolean_result_t<dyn, dyn>
boolean_of<dyn, dyn>(const dyn &, const dyn &, tf::boolean_op);

} // namespace tf::test
