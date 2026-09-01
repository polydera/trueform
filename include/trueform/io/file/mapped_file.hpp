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

#include <cstddef>
#include <limits>
#include <string>
#include <string_view>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define TF_IO_FILE_RESTORE_WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define TF_IO_FILE_RESTORE_NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#if defined(TF_IO_FILE_RESTORE_NOMINMAX)
#undef NOMINMAX
#undef TF_IO_FILE_RESTORE_NOMINMAX
#endif
#if defined(TF_IO_FILE_RESTORE_WIN32_LEAN_AND_MEAN)
#undef WIN32_LEAN_AND_MEAN
#undef TF_IO_FILE_RESTORE_WIN32_LEAN_AND_MEAN
#endif
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace tf::io {

/// @ingroup io
/// @brief Owns a read-only operating-system mapping of a non-empty file.
///
/// The mapping is move-only and releases all platform handles on destruction.
/// A false value indicates that the file could not be opened, sized, or mapped.
///
/// @pre The file is not modified while the mapping is alive.
class mapped_file {
public:
  mapped_file() = default;
  explicit mapped_file(std::string_view path) { open(path); }

  mapped_file(const mapped_file &) = delete;
  auto operator=(const mapped_file &) -> mapped_file & = delete;

  mapped_file(mapped_file &&other) noexcept { move_from(other); }
  auto operator=(mapped_file &&other) noexcept -> mapped_file & {
    if (this != &other) {
      close();
      move_from(other);
    }
    return *this;
  }

  ~mapped_file() { close(); }

  explicit operator bool() const { return _data != nullptr; }
  auto data() const -> const char * { return _data; }
  auto size() const -> std::size_t { return _size; }

private:
  auto open(std::string_view path) -> void {
    if (path.empty())
      return;
    const std::string filename(path);
#if defined(_WIN32)
    _file = CreateFileA(filename.c_str(), GENERIC_READ,
                        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (_file == INVALID_HANDLE_VALUE)
      return;
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(_file, &size) || size.QuadPart <= 0 ||
        static_cast<unsigned long long>(size.QuadPart) >
            (std::numeric_limits<std::size_t>::max)()) {
      close();
      return;
    }
    _size = static_cast<std::size_t>(size.QuadPart);
    _mapping = CreateFileMappingA(_file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (_mapping == nullptr) {
      close();
      return;
    }
    _data = static_cast<const char *>(
        MapViewOfFile(_mapping, FILE_MAP_READ, 0, 0, _size));
    if (_data == nullptr)
      close();
#else
#if defined(O_CLOEXEC)
    _file = ::open(filename.c_str(), O_RDONLY | O_CLOEXEC);
#else
    _file = ::open(filename.c_str(), O_RDONLY);
#endif
    if (_file < 0)
      return;
    struct stat status {};
    if (::fstat(_file, &status) != 0 || status.st_size <= 0 ||
        static_cast<unsigned long long>(status.st_size) >
            (std::numeric_limits<std::size_t>::max)()) {
      close();
      return;
    }
    _size = static_cast<std::size_t>(status.st_size);
    auto *mapping = ::mmap(nullptr, _size, PROT_READ, MAP_PRIVATE, _file, 0);
    if (mapping == MAP_FAILED) {
      close();
      return;
    }
    _data = static_cast<const char *>(mapping);
#endif
  }

  auto close() -> void {
#if defined(_WIN32)
    if (_data != nullptr)
      UnmapViewOfFile(_data);
    if (_mapping != nullptr)
      CloseHandle(_mapping);
    if (_file != INVALID_HANDLE_VALUE)
      CloseHandle(_file);
    _mapping = nullptr;
    _file = INVALID_HANDLE_VALUE;
#else
    if (_data != nullptr)
      ::munmap(const_cast<char *>(_data), _size);
    if (_file >= 0)
      ::close(_file);
    _file = -1;
#endif
    _data = nullptr;
    _size = 0;
  }

  auto move_from(mapped_file &other) -> void {
    _data = other._data;
    _size = other._size;
    other._data = nullptr;
    other._size = 0;
#if defined(_WIN32)
    _file = other._file;
    _mapping = other._mapping;
    other._file = INVALID_HANDLE_VALUE;
    other._mapping = nullptr;
#else
    _file = other._file;
    other._file = -1;
#endif
  }

  const char *_data = nullptr;
  std::size_t _size = 0;
#if defined(_WIN32)
  HANDLE _file = INVALID_HANDLE_VALUE;
  HANDLE _mapping = nullptr;
#else
  int _file = -1;
#endif
};

} // namespace tf::io
