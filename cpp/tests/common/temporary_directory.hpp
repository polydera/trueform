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

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

namespace tf::cpp::test {

class temporary_directory {
  std::filesystem::path _path;

  static auto safe_prefix(std::string prefix) -> std::string {
    for (auto &character : prefix) {
      const auto value = static_cast<unsigned char>(character);
      if (!std::isalnum(value) && character != '-' && character != '_')
        character = '_';
    }
    return prefix.empty() ? std::string("trueform-cpp") : prefix;
  }

  static auto candidate_name(const std::string &prefix) -> std::string {
    static std::atomic<std::uint64_t> sequence{0};
    std::random_device random;
    const auto random_bits =
        (static_cast<std::uint64_t>(random()) << 32U) ^ random();
    const auto ticks = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const auto thread = static_cast<std::uint64_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id()));
    const auto serial = sequence.fetch_add(1, std::memory_order_relaxed);

    std::ostringstream name;
    name << prefix << '-' << std::hex << ticks << '-' << thread << '-'
         << random_bits << '-' << serial;
    return name.str();
  }

  auto cleanup() noexcept -> void {
    if (_path.empty())
      return;
    std::error_code error;
    std::filesystem::remove_all(_path, error);
    _path.clear();
  }

public:
  explicit temporary_directory(std::string prefix = "trueform-cpp-test") {
    const auto root = std::filesystem::temp_directory_path();
    const auto normalized = safe_prefix(std::move(prefix));
    for (int attempt = 0; attempt < 128; ++attempt) {
      auto candidate = root / candidate_name(normalized);
      std::error_code error;
      if (std::filesystem::create_directory(candidate, error)) {
        _path = std::move(candidate);
        return;
      }
      if (error && error != std::errc::file_exists)
        throw std::filesystem::filesystem_error(
            "unable to create temporary directory", candidate, error);
    }
    throw std::runtime_error("unable to allocate a unique temporary directory");
  }

  temporary_directory(const temporary_directory &) = delete;
  auto operator=(const temporary_directory &) -> temporary_directory & = delete;

  temporary_directory(temporary_directory &&other) noexcept
      : _path(std::move(other._path)) {
    other._path.clear();
  }

  auto operator=(temporary_directory &&other) noexcept
      -> temporary_directory & {
    if (this != &other) {
      cleanup();
      _path = std::move(other._path);
      other._path.clear();
    }
    return *this;
  }

  ~temporary_directory() { cleanup(); }

  auto path() const noexcept -> const std::filesystem::path & { return _path; }
};

} // namespace tf::cpp::test
