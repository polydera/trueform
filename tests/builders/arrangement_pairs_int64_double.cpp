/**
 * @file arrangement_pairs_int64_double.cpp
 * @brief The compiled two-operand arrangement tier for int64 indices, double
 * coordinates
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "arrangement_builders_impl.hpp"

#include <cstdint>

namespace tf::test {
namespace {
using tri = form_t<std::int64_t, double, 3>;
using quad = form_t<std::int64_t, double, 4>;
using dyn = form_t<std::int64_t, double, tf::dynamic_size>;
using tri_forms = forms_range_t<std::int64_t, double, 3>;
using quad_forms = forms_range_t<std::int64_t, double, 4>;
using dyn_forms = forms_range_t<std::int64_t, double, tf::dynamic_size>;
} // namespace

template pair_arrangement_t<tri, tri>
build_pair_arrangement<tri, tri>(const tri &, const tri &,
                                 tf::arrangement_config);

template pair_arrangement_t<tri, quad>
build_pair_arrangement<tri, quad>(const tri &, const quad &,
                                  tf::arrangement_config);

template pair_arrangement_t<tri, dyn>
build_pair_arrangement<tri, dyn>(const tri &, const dyn &,
                                 tf::arrangement_config);

template pair_arrangement_t<quad, quad>
build_pair_arrangement<quad, quad>(const quad &, const quad &,
                                   tf::arrangement_config);

template pair_arrangement_t<quad, dyn>
build_pair_arrangement<quad, dyn>(const quad &, const dyn &,
                                  tf::arrangement_config);

template pair_arrangement_t<dyn, tri>
build_pair_arrangement<dyn, tri>(const dyn &, const tri &,
                                 tf::arrangement_config);

template pair_arrangement_t<dyn, quad>
build_pair_arrangement<dyn, quad>(const dyn &, const quad &,
                                  tf::arrangement_config);

template pair_arrangement_t<dyn, dyn>
build_pair_arrangement<dyn, dyn>(const dyn &, const dyn &,
                                 tf::arrangement_config);

} // namespace tf::test
