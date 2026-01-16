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
#pragma once
#include <iterator>

namespace tf::iter {

template <typename Derived, typename Handle>
struct forward_stride_base_api : Handle {

public:
  forward_stride_base_api() = default;
  forward_stride_base_api(Handle h) : Handle{std::move(h)} {}

  auto operator*() const -> decltype(auto) { return Handle::dereference(); }

  auto operator++() -> Derived & {
    Handle::base_iter() =
        std::next(Handle::base_iter(), Handle::iterator_stride());
    return static_cast<Derived &>(*this);
  }

  auto operator++(int) -> Derived {
    auto out = as_derived();
    ++(*this);
    return out;
  }

  friend auto operator==(const Derived &obj, const Derived &other) {
    return obj.base_iter() == other.base_iter();
  }
  friend auto operator!=(const Derived &obj, const Derived &other) {
    return obj.base_iter() != other.base_iter();
  }

protected:
  auto as_derived() -> Derived & { return static_cast<Derived &>(*this); }
  auto as_derived() const -> const Derived & {
    return static_cast<const Derived &>(*this);
  }
};

template <typename Derived, typename Handle>
struct bidirectional_stride_base_api
    : forward_stride_base_api<Derived, Handle> {
private:
  using base_t = forward_stride_base_api<Derived, Handle>;

public:
  using base_t::base_t;
  auto operator--() -> Derived & {
    Handle::base_iter() =
        std::prev(Handle::base_iter(), Handle::iterator_stride());
    return static_cast<Derived &>(*this);
  }

  auto operator--(int) -> Derived {
    auto out = base_t::as_derived();
    --(*this);
    return out;
  }
};

template <typename Derived, typename Handle>
struct random_access_stride_base_api
    : bidirectional_stride_base_api<Derived, Handle> {
private:
  using base_t = bidirectional_stride_base_api<Derived, Handle>;

public:
  using base_t::base_t;
  auto operator+=(typename base_t::difference_type n) -> Derived & {

    Handle::base_iter() += n * Handle::iterator_stride();
    return static_cast<Derived &>(*this);
  }

  auto operator+(typename base_t::difference_type n) const -> Derived {
    auto out = static_cast<const Derived &>(*this);
    out += n;
    return out;
  }

  friend auto operator+(typename base_t::difference_type n, const Derived &it) {
    return it + n;
  }

  auto operator-=(typename base_t::difference_type n) -> Derived & {
    Handle::base_iter() -= n * Handle::iterator_stride();
    return static_cast<Derived &>(*this);
  }

  auto operator-(typename base_t::difference_type n) const -> Derived {
    auto out = static_cast<const Derived &>(*this);
    out -= n;
    return out;
  }

  friend auto operator-(const Derived &obj, const Derived &other)
      -> typename base_t::difference_type {
    return (obj.base_iter() - other.base_iter()) / obj.iterator_stride();
  }

  auto operator[](typename base_t::difference_type n) const -> decltype(auto) {
    return *(base_t::as_derived() + n);
  }
  auto operator[](typename base_t::difference_type n) -> decltype(auto) {
    return *(base_t::as_derived() + n);
  }

  friend auto operator<(const Derived &obj, const Derived &other) {
    return obj.base_iter() < other.base_iter();
  }
  friend auto operator<=(const Derived &obj, const Derived &other) {
    return obj.base_iter() <= other.base_iter();
  }
  friend auto operator>(const Derived &obj, const Derived &other) {
    return obj.base_iter() > other.base_iter();
  }
  friend auto operator>=(const Derived &obj, const Derived &other) {
    return obj.base_iter() >= other.base_iter();
  }
};

template <typename Category> struct stride_base_api {
  template <typename Derived, typename, typename Handle>
  using type = forward_stride_base_api<Derived, Handle>;
};

template <> struct stride_base_api<std::bidirectional_iterator_tag> {
  template <typename Derived, typename, typename Handle>
  using type = bidirectional_stride_base_api<Derived, Handle>;
};

template <> struct stride_base_api<std::random_access_iterator_tag> {
  template <typename Derived, typename Handle>
  using type = random_access_stride_base_api<Derived, Handle>;
};

template <typename Derived, typename Handle>
struct stride_api
    : stride_base_api<typename Handle::iterator_category>::template type<
          Derived, Handle> {
private:
  using base_t = typename stride_base_api<
      typename Handle::iterator_category>::template type<Derived, Handle>;

public:
  using base_t::base_t;
};
} // namespace tf::iter
