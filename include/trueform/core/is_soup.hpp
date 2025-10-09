#pragma once
#include "./base/soup.hpp"

namespace tf {
namespace core {
constexpr auto is_soup(const void *) -> std::false_type;

template <typename T>
constexpr auto is_soup(const tf::core::soup<T> *) -> std::true_type;
} // namespace core
template <typename Policy>
inline constexpr bool is_soup =
    decltype(core::is_soup(static_cast<const Policy *>(nullptr)))::value;
} // namespace tf
