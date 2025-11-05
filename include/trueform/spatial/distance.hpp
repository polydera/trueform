/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./neighbor_search.hpp"

namespace tf {
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::form<Dims, Policy0> &form,
               const tf::point_like<Dims, Policy1> &obj) {
  return neighbor_search(form, obj).metric();
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::point_like<Dims, Policy0> &obj,
               const tf::form<Dims, Policy0> &form) {
  return neighbor_search(form, obj).metric();
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::form<Dims, Policy0> &form,
              const tf::point_like<Dims, Policy1> &obj) {
  return tf::sqrt(distance2(form, obj));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::point_like<Dims, Policy0> &obj,
              const tf::form<Dims, Policy0> &form) {
  return tf::sqrt(distance2(form, obj));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::form<Dims, Policy0> &form,
               const tf::segment<Dims, Policy1> &obj) {
  return neighbor_search(form, obj).metric();
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::segment<Dims, Policy0> &obj,
               const tf::form<Dims, Policy0> &form) {
  return neighbor_search(form, obj).metric();
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::form<Dims, Policy0> &form,
              const tf::segment<Dims, Policy1> &obj) {
  return tf::sqrt(distance2(form, obj));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::segment<Dims, Policy0> &obj,
              const tf::form<Dims, Policy0> &form) {
  return tf::sqrt(distance2(form, obj));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::form<Dims, Policy0> &form,
               const tf::line_like<Dims, Policy1> &obj) {
  return neighbor_search(form, obj).metric();
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::line_like<Dims, Policy0> &obj,
               const tf::form<Dims, Policy0> &form) {
  return neighbor_search(form, obj).metric();
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::form<Dims, Policy0> &form,
              const tf::line_like<Dims, Policy1> &obj) {
  return tf::sqrt(distance2(form, obj));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::line_like<Dims, Policy0> &obj,
              const tf::form<Dims, Policy0> &form) {
  return tf::sqrt(distance2(form, obj));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::form<Dims, Policy0> &form,
               const tf::ray_like<Dims, Policy1> &obj) {
  return neighbor_search(form, obj).metric();
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::ray_like<Dims, Policy0> &obj,
               const tf::form<Dims, Policy0> &form) {
  return neighbor_search(form, obj).metric();
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::form<Dims, Policy0> &form,
              const tf::ray_like<Dims, Policy1> &obj) {
  return tf::sqrt(distance2(form, obj));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::ray_like<Dims, Policy0> &obj,
              const tf::form<Dims, Policy0> &form) {
  return tf::sqrt(distance2(form, obj));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::form<Dims, Policy0> &form,
               const tf::polygon<Dims, Policy1> &obj) {
  return neighbor_search(form, obj).metric();
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::polygon<Dims, Policy0> &obj,
               const tf::form<Dims, Policy0> &form) {
  return neighbor_search(form, obj).metric();
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::form<Dims, Policy0> &form,
              const tf::polygon<Dims, Policy1> &obj) {
  return tf::sqrt(distance2(form, obj));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::polygon<Dims, Policy0> &obj,
              const tf::form<Dims, Policy0> &form) {
  return tf::sqrt(distance2(form, obj));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::form<Dims, Policy0> &form0,
               const tf::form<Dims, Policy1> &form1) {
  return neighbor_search(form0, form1).metric();
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::form<Dims, Policy0> &form0,
              const tf::form<Dims, Policy1> &form1) {
  return tf::sqrt(distance2(form0, form1));
}

} // namespace tf
