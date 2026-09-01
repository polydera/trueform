/**
 * @file arrangement_builders_int32_double.cpp
 * @brief The compiled one- and N-operand arrangement tier for int32 indices,
 * double coordinates
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "arrangement_builders_impl.hpp"
#include "arrangement_readers_impl.hpp"

#include <cstdint>

namespace tf::test {
namespace {
using tri = form_t<std::int32_t, double, 3>;
using quad = form_t<std::int32_t, double, 4>;
using dyn = form_t<std::int32_t, double, tf::dynamic_size>;
using tri_forms = forms_range_t<std::int32_t, double, 3>;
using quad_forms = forms_range_t<std::int32_t, double, 4>;
using dyn_forms = forms_range_t<std::int32_t, double, tf::dynamic_size>;
} // namespace

template self_arrangement_t<tri>
build_self_arrangement<tri>(const tri &, tf::arrangement_config);

template self_arrangement_t<quad>
build_self_arrangement<quad>(const quad &, tf::arrangement_config);

template self_arrangement_t<dyn>
build_self_arrangement<dyn>(const dyn &, tf::arrangement_config);

template range_arrangement_t<tri_forms>
    build_range_arrangement<tri_forms>(tri_forms, tf::arrangement_config);

template range_arrangement_t<quad_forms>
    build_range_arrangement<quad_forms>(quad_forms, tf::arrangement_config);

template range_arrangement_t<dyn_forms>
    build_range_arrangement<dyn_forms>(dyn_forms, tf::arrangement_config);

template polygon_arrangements_result_t<tri>
polygon_arrangements_of<tri>(const tri &, tf::arrangement_config);

template polygon_arrangements_curves_result_t<tri>
polygon_arrangements_with_curves_of<tri>(const tri &, tf::arrangement_config);

template polygon_arrangements_result_t<quad>
polygon_arrangements_of<quad>(const quad &, tf::arrangement_config);

template polygon_arrangements_curves_result_t<quad>
polygon_arrangements_with_curves_of<quad>(const quad &, tf::arrangement_config);

template polygon_arrangements_result_t<dyn>
polygon_arrangements_of<dyn>(const dyn &, tf::arrangement_config);

template polygon_arrangements_curves_result_t<dyn>
polygon_arrangements_with_curves_of<dyn>(const dyn &, tf::arrangement_config);

template mesh_arrangements_result_t<tri_forms>
    mesh_arrangements_of<tri_forms>(tri_forms, tf::arrangement_config);

template mesh_arrangements_curves_result_t<tri_forms>
    mesh_arrangements_with_curves_of<tri_forms>(tri_forms,
                                                tf::arrangement_config);

template mesh_arrangements_result_t<quad_forms>
    mesh_arrangements_of<quad_forms>(quad_forms, tf::arrangement_config);

template mesh_arrangements_curves_result_t<quad_forms>
    mesh_arrangements_with_curves_of<quad_forms>(quad_forms,
                                                 tf::arrangement_config);

template mesh_arrangements_result_t<dyn_forms>
    mesh_arrangements_of<dyn_forms>(dyn_forms, tf::arrangement_config);

template mesh_arrangements_curves_result_t<dyn_forms>
    mesh_arrangements_with_curves_of<dyn_forms>(dyn_forms,
                                                tf::arrangement_config);

} // namespace tf::test
