/**
 * @file arrangement_reads_int64_double.cpp
 * @brief The compiled arrangement read tier for int64 indices, double
 * coordinates
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "arrangement_builders_impl.hpp"
#include "arrangement_readers_impl.hpp"

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

template arrangement_curves_t<self_arrangement_t<tri>>
arrangement_curves_of<self_arrangement_t<tri>>(const self_arrangement_t<tri> &);

template arrangement_mesh_t<self_arrangement_t<tri>>
arrangement_mesh_of<self_arrangement_t<tri>>(const self_arrangement_t<tri> &);

template arrangement_curves_t<self_arrangement_t<quad>>
arrangement_curves_of<self_arrangement_t<quad>>(
    const self_arrangement_t<quad> &);

template arrangement_mesh_t<self_arrangement_t<quad>>
arrangement_mesh_of<self_arrangement_t<quad>>(const self_arrangement_t<quad> &);

template arrangement_curves_t<self_arrangement_t<dyn>>
arrangement_curves_of<self_arrangement_t<dyn>>(const self_arrangement_t<dyn> &);

template arrangement_mesh_t<self_arrangement_t<dyn>>
arrangement_mesh_of<self_arrangement_t<dyn>>(const self_arrangement_t<dyn> &);

template arrangement_curves_t<range_arrangement_t<tri_forms>>
arrangement_curves_of<range_arrangement_t<tri_forms>>(
    const range_arrangement_t<tri_forms> &);

template arrangement_mesh_t<range_arrangement_t<tri_forms>>
arrangement_mesh_of<range_arrangement_t<tri_forms>>(
    const range_arrangement_t<tri_forms> &);

template arrangement_curves_t<range_arrangement_t<quad_forms>>
arrangement_curves_of<range_arrangement_t<quad_forms>>(
    const range_arrangement_t<quad_forms> &);

template arrangement_mesh_t<range_arrangement_t<quad_forms>>
arrangement_mesh_of<range_arrangement_t<quad_forms>>(
    const range_arrangement_t<quad_forms> &);

template arrangement_curves_t<range_arrangement_t<dyn_forms>>
arrangement_curves_of<range_arrangement_t<dyn_forms>>(
    const range_arrangement_t<dyn_forms> &);

template arrangement_mesh_t<range_arrangement_t<dyn_forms>>
arrangement_mesh_of<range_arrangement_t<dyn_forms>>(
    const range_arrangement_t<dyn_forms> &);

template arrangement_curves_t<pair_arrangement_t<tri, tri>>
arrangement_curves_of<pair_arrangement_t<tri, tri>>(
    const pair_arrangement_t<tri, tri> &);

template arrangement_mesh_t<pair_arrangement_t<tri, tri>>
arrangement_mesh_of<pair_arrangement_t<tri, tri>>(
    const pair_arrangement_t<tri, tri> &);

template arrangement_curves_t<pair_arrangement_t<tri, quad>>
arrangement_curves_of<pair_arrangement_t<tri, quad>>(
    const pair_arrangement_t<tri, quad> &);

template arrangement_mesh_t<pair_arrangement_t<tri, quad>>
arrangement_mesh_of<pair_arrangement_t<tri, quad>>(
    const pair_arrangement_t<tri, quad> &);

template arrangement_curves_t<pair_arrangement_t<tri, dyn>>
arrangement_curves_of<pair_arrangement_t<tri, dyn>>(
    const pair_arrangement_t<tri, dyn> &);

template arrangement_mesh_t<pair_arrangement_t<tri, dyn>>
arrangement_mesh_of<pair_arrangement_t<tri, dyn>>(
    const pair_arrangement_t<tri, dyn> &);

template arrangement_curves_t<pair_arrangement_t<quad, quad>>
arrangement_curves_of<pair_arrangement_t<quad, quad>>(
    const pair_arrangement_t<quad, quad> &);

template arrangement_mesh_t<pair_arrangement_t<quad, quad>>
arrangement_mesh_of<pair_arrangement_t<quad, quad>>(
    const pair_arrangement_t<quad, quad> &);

template arrangement_curves_t<pair_arrangement_t<quad, dyn>>
arrangement_curves_of<pair_arrangement_t<quad, dyn>>(
    const pair_arrangement_t<quad, dyn> &);

template arrangement_mesh_t<pair_arrangement_t<quad, dyn>>
arrangement_mesh_of<pair_arrangement_t<quad, dyn>>(
    const pair_arrangement_t<quad, dyn> &);

template arrangement_curves_t<pair_arrangement_t<dyn, tri>>
arrangement_curves_of<pair_arrangement_t<dyn, tri>>(
    const pair_arrangement_t<dyn, tri> &);

template arrangement_mesh_t<pair_arrangement_t<dyn, tri>>
arrangement_mesh_of<pair_arrangement_t<dyn, tri>>(
    const pair_arrangement_t<dyn, tri> &);

template arrangement_curves_t<pair_arrangement_t<dyn, quad>>
arrangement_curves_of<pair_arrangement_t<dyn, quad>>(
    const pair_arrangement_t<dyn, quad> &);

template arrangement_mesh_t<pair_arrangement_t<dyn, quad>>
arrangement_mesh_of<pair_arrangement_t<dyn, quad>>(
    const pair_arrangement_t<dyn, quad> &);

template arrangement_curves_t<pair_arrangement_t<dyn, dyn>>
arrangement_curves_of<pair_arrangement_t<dyn, dyn>>(
    const pair_arrangement_t<dyn, dyn> &);

template arrangement_mesh_t<pair_arrangement_t<dyn, dyn>>
arrangement_mesh_of<pair_arrangement_t<dyn, dyn>>(
    const pair_arrangement_t<dyn, dyn> &);

} // namespace tf::test
