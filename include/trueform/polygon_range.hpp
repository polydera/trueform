/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./implementation/mapped_iterator.hpp"
#include "./polygon.hpp"
namespace tf {
namespace implementation {
template <typename Range0> struct polygon_range_policy {
  Range0 points;
  template <typename Range> auto operator()(Range &&ids) const {
    return tf::make_polygon(ids, points);
  }

  template <typename Iterator0, typename Iterator1>
  auto operator()(std::pair<Iterator0, Iterator1> iters) const {
    return tf::make_polygon(*iters.first, points, *iters.second);
  }
};

template <typename Iterator0, typename Range1, typename Iterator1>
auto make_polygon_range_iter(Iterator0 faces_iter, Range1 &&points,
                             Iterator1 normals_iter) {
  auto begins = std::make_pair(faces_iter, normals_iter);
  return iter::make_iter_mapped(
      begins, polygon_range_policy<std::decay_t<Range1>>{points});
}

} // namespace implementation

template <typename Range0, typename Range1, typename...> struct polygon_range {
  using iterator = decltype(tf::implementation::iter::make_mapped(
      std::declval<const Range0>().begin(),
      implementation::polygon_range_policy<Range1>{
          tf::make_range(std::declval<const Range1>())}));
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using pointer = typename std::iterator_traits<iterator>::pointer;
  using const_iterator = iterator;
  using size_type = std::size_t;

  polygon_range(const Range0 &faces, const Range1 &points)
      : _faces(faces), _points{points} {}

  auto faces() const -> const Range0 & { return _faces; }

  auto points() const -> const Range1 & { return _points; }

  auto begin() const {
    return tf::implementation::iter::make_mapped(
        _faces.begin(),
        implementation::polygon_range_policy<Range1>{tf::make_range(_points)});
  }

  auto end() const {
    return tf::implementation::iter::make_mapped(
        _faces.end(),
        implementation::polygon_range_policy<Range1>{tf::make_range(_points)});
  }

  auto size() const -> size_type { return _faces.size(); }
  auto empty() const -> bool { return _faces.size() == 0; }

  auto front() const -> reference { return *begin(); }
  auto back() const -> reference { return *(begin() + size() - 1); }
  auto operator[](size_type i) const -> reference { return *(begin() + i); }

private:
  Range0 _faces;
  Range1 _points;
};

template <typename Range0, typename Range1, typename Range2>
struct polygon_range<Range0, Range1, Range2> {
  using iterator = decltype(tf::implementation::make_polygon_range_iter(
      std::declval<const Range0>().begin(), std::declval<const Range1>(),
      std::declval<const Range2>().begin()));
  using value_type = typename std::iterator_traits<iterator>::value_type;
  using reference = typename std::iterator_traits<iterator>::reference;
  using pointer = typename std::iterator_traits<iterator>::pointer;
  using const_iterator = iterator;
  using size_type = std::size_t;

  polygon_range(const Range0 &faces, const Range1 &points,
                const Range2 &normals)
      : _faces(faces), _points{points}, _normals{normals} {}

  auto faces() const -> const Range0 & { return _faces; }

  auto points() const -> const Range1 & { return _points; }

  auto normals() const -> const Range2 & { return _normals; }

  auto begin() const {
    return implementation::make_polygon_range_iter(_faces.begin(), _normals,
                                                   _normals.begin());
  }

  auto end() const {
    return implementation::make_polygon_range_iter(_faces.end(), _normals,
                                                   _normals.end());
  }

  auto size() const -> size_type { return _faces.size(); }
  auto empty() const -> bool { return _faces.size() == 0; }

  auto front() const -> reference { return *begin(); }
  auto back() const -> reference { return *(begin() + size() - 1); }
  auto operator[](size_type i) const -> reference { return *(begin() + i); }

private:
  Range0 _faces;
  Range1 _points;
  Range2 _normals;
};
} // namespace tf
