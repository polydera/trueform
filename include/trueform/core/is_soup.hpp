#pragma once
#include "./polygons.hpp"
#include "./segments.hpp"

namespace tf {
namespace core {
template <typename T> constexpr auto is_soup(const void*) -> bool {
  return false;
}
template <typename T>
constexpr auto is_soup(const tf::core::soup<T> *) -> bool {
  return true;
}
} // namespace core
template <typename Policy>
constexpr auto is_soup(const tf::polygons<Policy> &obj) -> bool {
  return core::is_soup(&obj);
}
template <typename Policy>
constexpr auto is_soup(const tf::segments<Policy> &obj) -> bool {
  return core::is_soup(&obj);
}
} // namespace tf
