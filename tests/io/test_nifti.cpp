/**
 * @file test_nifti.cpp
 * @brief Tests for NIfTI-1 reading and writing
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <atomic>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <type_traits>
#include <trueform/core.hpp>
#include <trueform/io.hpp>

namespace {

auto nifti_temp_path(const char *extension) -> std::filesystem::path {
  static std::atomic<int> counter{0};
  static auto process_id = std::random_device{}();
  auto id = counter.fetch_add(1);
  auto name = std::string("trueform_nifti_test_") +
              std::to_string(process_id) + "_" + std::to_string(id) +
              extension;
  return std::filesystem::temp_directory_path() / name;
}

struct nifti_temp_file_cleanup {
  std::filesystem::path path;
  ~nifti_temp_file_cleanup() { std::filesystem::remove(path); }
};

template <typename T>
auto nifti_test_volume() -> tf::volume_buffer<T, float, 3> {
  tf::volume_buffer<T, float, 3> volume({5, 4, 3}, {0.7f, 0.7f, 1.5f},
                                        {1.f, -2.f, 3.f});
  for (std::size_t i = 0; i < volume.voxel_count(); ++i) {
    if constexpr (std::is_unsigned_v<T>)
      volume.samples_buffer().begin()[i] = T(i % 97);
    else
      volume.samples_buffer().begin()[i] = T(double(i % 97) - 41.0);
  }
  return volume;
}

auto nifti_copy(const tf::buffer<char> &bytes) -> tf::buffer<char> {
  tf::buffer<char> out;
  out.allocate(bytes.size());
  std::memcpy(out.begin(), bytes.begin(), bytes.size());
  return out;
}

auto nifti_bytes_range(const tf::buffer<char> &bytes)
    -> tf::range<const char *, tf::dynamic_size> {
  return tf::make_range(static_cast<const char *>(bytes.begin()),
                        bytes.size());
}

template <typename A, typename B>
auto nifti_samples_equal(const A &a, const B &b) -> bool {
  if (a.voxel_count() != b.voxel_count())
    return false;
  bool equal = true;
  for (std::size_t i = 0; i < a.voxel_count(); ++i)
    equal = equal && double(a.samples_buffer().begin()[i]) ==
                         double(b.samples_buffer().begin()[i]);
  return equal;
}

auto nifti_swap_at(char *bytes, std::size_t offset, std::size_t width)
    -> void {
  for (std::size_t i = 0; i < width / 2; ++i) {
    const auto keep = bytes[offset + i];
    bytes[offset + i] = bytes[offset + width - 1 - i];
    bytes[offset + width - 1 - i] = keep;
  }
}

} // namespace

TEMPLATE_TEST_CASE("nifti: a volume round-trips bit-exact in its own type",
                   "[io][nifti]", std::uint8_t, std::int16_t, std::uint16_t,
                   std::int32_t, float, double) {
  const auto volume = nifti_test_volume<TestType>();
  const auto bytes = tf::write_nifti_to_buffer(volume);
  const auto file = tf::read_nifti<TestType>(nifti_bytes_range(bytes));
  REQUIRE(bool(file));
  REQUIRE(file.volume.dims() == volume.dims());
  REQUIRE_FALSE(file.posed);
  for (std::size_t axis = 0; axis < 3; ++axis) {
    REQUIRE(file.volume.spacing()[axis] == volume.spacing()[axis]);
    REQUIRE(file.volume.origin()[axis] == volume.origin()[axis]);
  }
  REQUIRE(nifti_samples_equal(volume, file.volume));
}

TEST_CASE("nifti: files round-trip through .nii and .nii.gz",
          "[io][nifti]") {
  const auto volume = nifti_test_volume<std::int16_t>();
  for (const auto *extension : {".nii", ".nii.gz"}) {
    const auto path = nifti_temp_path(extension);
    nifti_temp_file_cleanup cleanup{path};
    REQUIRE(tf::write_nifti(volume, path.string()));
    const auto file = tf::read_nifti<std::int16_t>(path.string());
    REQUIRE(bool(file));
    REQUIRE(file.volume.dims() == volume.dims());
    REQUIRE(nifti_samples_equal(volume, file.volume));
  }
}

TEST_CASE("nifti: a stated type converts the file's samples", "[io][nifti]") {
  const auto volume = nifti_test_volume<std::int16_t>();
  const auto bytes = tf::write_nifti_to_buffer(volume);
  const auto file = tf::read_nifti<float>(nifti_bytes_range(bytes));
  REQUIRE(bool(file));
  REQUIRE(nifti_samples_equal(volume, file.volume));
}

TEST_CASE("nifti: scl scaling is applied on read", "[io][nifti]") {
  const auto volume = nifti_test_volume<std::int16_t>();
  auto bytes = tf::write_nifti_to_buffer(volume);
  const float slope = 2.5f;
  const float intercept = -3.f;
  std::memcpy(bytes.begin() + 112, &slope, 4);
  std::memcpy(bytes.begin() + 116, &intercept, 4);
  const auto file = tf::read_nifti<float>(nifti_bytes_range(bytes));
  REQUIRE(bool(file));
  const auto info = tf::read_nifti_header(nifti_bytes_range(bytes));
  REQUIRE(info.slope == slope);
  REQUIRE(info.intercept == intercept);
  for (std::size_t i = 0; i < volume.voxel_count(); ++i)
    REQUIRE(file.volume.samples_buffer().begin()[i] ==
            float(double(slope) *
                      double(volume.samples_buffer().begin()[i]) +
                  double(intercept)));
}

TEST_CASE("nifti: the header's facts read without the samples",
          "[io][nifti]") {
  const auto volume = nifti_test_volume<std::uint16_t>();
  const auto bytes = tf::write_nifti_to_buffer(volume);
  const auto info = tf::read_nifti_header(nifti_bytes_range(bytes));
  REQUIRE(bool(info));
  REQUIRE(info.datatype == tf::nifti_datatype::uint16);
  REQUIRE(info.dims == volume.dims());
  REQUIRE_FALSE(info.posed);
  for (std::size_t axis = 0; axis < 3; ++axis)
    REQUIRE(info.spacing[axis] == volume.spacing()[axis]);

  const auto gz = tf::io::gzip_deflated(nifti_bytes_range(bytes));
  const auto gz_info = tf::read_nifti_header(nifti_bytes_range(gz));
  REQUIRE(bool(gz_info));
  REQUIRE(gz_info.datatype == tf::nifti_datatype::uint16);
  REQUIRE(gz_info.dims == volume.dims());
}

TEST_CASE("nifti: a posed file keeps its frame, an aligned one its origin",
          "[io][nifti]") {
  const auto volume = nifti_test_volume<float>();
  auto world = tf::make_identity_transformation<float, 3>();
  world(0, 0) = 0.f;
  world(0, 1) = -1.f;
  world(1, 0) = 1.f;
  world(1, 1) = 0.f;
  world(0, 3) = 5.f;
  world(1, 3) = 6.f;
  world(2, 3) = 7.f;
  const auto frame = tf::make_frame(world);

  const auto bytes = tf::write_nifti_to_buffer(volume, frame);
  const auto file = tf::read_nifti<float>(nifti_bytes_range(bytes));
  REQUIRE(bool(file));
  REQUIRE(file.posed);
  // The file holds one affine, so the read is the canonical split: the
  // rotation in the frame, the whole translation with it, the grid origin
  // zero. The physical map is what round-trips.
  for (std::size_t r = 0; r < 3; ++r)
    for (std::size_t c = 0; c < 3; ++c)
      REQUIRE(std::abs(file.frame.transformation()(r, c) - world(r, c)) <
              1e-5f);
  for (std::size_t axis = 0; axis < 3; ++axis) {
    REQUIRE(std::abs(file.volume.spacing()[axis] - volume.spacing()[axis]) <
            1e-6f);
    REQUIRE(file.volume.origin()[axis] == 0.f);
  }
  for (const auto &corner :
       {std::array<int, 3>{0, 0, 0}, std::array<int, 3>{4, 3, 2}}) {
    const auto written = tf::transformed(
        volume.point_at(corner[0], corner[1], corner[2]), frame);
    const auto read = tf::transformed(
        file.volume.point_at(corner[0], corner[1], corner[2]), file.frame);
    for (std::size_t axis = 0; axis < 3; ++axis)
      REQUIRE(std::abs(written[axis] - read[axis]) < 1e-4f);
  }

  const auto aligned = tf::write_nifti_to_buffer(volume);
  const auto aligned_file = tf::read_nifti<float>(nifti_bytes_range(aligned));
  REQUIRE(bool(aligned_file));
  REQUIRE_FALSE(aligned_file.posed);
  for (std::size_t axis = 0; axis < 3; ++axis)
    REQUIRE(aligned_file.volume.origin()[axis] == volume.origin()[axis]);
}

TEST_CASE("nifti: a malformed file refuses with its fact named",
          "[io][nifti]") {
  const auto volume = nifti_test_volume<std::int16_t>();
  const auto bytes = tf::write_nifti_to_buffer(volume);

  SECTION("truncated") {
    const auto head = tf::make_range(
        static_cast<const char *>(bytes.begin()), std::size_t{100});
    REQUIRE(tf::read_nifti<float>(head).status ==
            tf::nifti_status::truncated);
    const auto file = tf::read_nifti<float>(tf::make_range(
        static_cast<const char *>(bytes.begin()), bytes.size() - 10));
    REQUIRE(file.status == tf::nifti_status::truncated);
  }
  SECTION("bad magic") {
    auto broken = nifti_copy(bytes);
    broken.begin()[344] = 'x';
    REQUIRE(tf::read_nifti<float>(nifti_bytes_range(broken)).status ==
            tf::nifti_status::bad_magic);
  }
  SECTION("header pair") {
    auto pair = nifti_copy(bytes);
    pair.begin()[344] = 'n';
    pair.begin()[345] = 'i';
    pair.begin()[346] = '1';
    REQUIRE(tf::read_nifti<float>(nifti_bytes_range(pair)).status ==
            tf::nifti_status::header_pair);
  }
  SECTION("unsupported datatype") {
    auto complex = nifti_copy(bytes);
    const std::int16_t code = 32;
    std::memcpy(complex.begin() + 70, &code, 2);
    REQUIRE(tf::read_nifti<float>(nifti_bytes_range(complex)).status ==
            tf::nifti_status::unsupported_datatype);
  }
  SECTION("a real fourth extent") {
    auto series = nifti_copy(bytes);
    const std::int16_t ndim = 4;
    const std::int16_t frames = 2;
    std::memcpy(series.begin() + 40, &ndim, 2);
    std::memcpy(series.begin() + 48, &frames, 2);
    REQUIRE(tf::read_nifti<float>(nifti_bytes_range(series)).status ==
            tf::nifti_status::unsupported_dims);
  }
  SECTION("missing file") {
    REQUIRE(tf::read_nifti<float>("no_such_trueform_nifti.nii").status ==
            tf::nifti_status::unreadable);
  }
}

TEST_CASE("nifti: an opposite-endian file reads through the swap",
          "[io][nifti]") {
  const auto volume = nifti_test_volume<std::int16_t>();
  auto bytes = tf::write_nifti_to_buffer(volume);
  auto *raw = bytes.begin();
  nifti_swap_at(raw, 0, 4);
  for (std::size_t i = 0; i < 8; ++i)
    nifti_swap_at(raw, 40 + 2 * i, 2);
  nifti_swap_at(raw, 70, 2);
  nifti_swap_at(raw, 72, 2);
  for (std::size_t i = 0; i < 8; ++i)
    nifti_swap_at(raw, 76 + 4 * i, 4);
  nifti_swap_at(raw, 108, 4);
  nifti_swap_at(raw, 112, 4);
  nifti_swap_at(raw, 116, 4);
  nifti_swap_at(raw, 252, 2);
  nifti_swap_at(raw, 254, 2);
  for (std::size_t i = 0; i < 6; ++i)
    nifti_swap_at(raw, 256 + 4 * i, 4);
  for (std::size_t i = 0; i < 12; ++i)
    nifti_swap_at(raw, 280 + 4 * i, 4);
  for (std::size_t i = 0; i < volume.voxel_count(); ++i)
    nifti_swap_at(raw, 352 + 2 * i, 2);

  const auto file = tf::read_nifti<std::int16_t>(nifti_bytes_range(bytes));
  REQUIRE(bool(file));
  REQUIRE(file.volume.dims() == volume.dims());
  REQUIRE(nifti_samples_equal(volume, file.volume));
}

namespace {

auto nifti_put_f32(tf::buffer<char> &bytes, std::size_t offset, float value)
    -> void {
  std::memcpy(bytes.begin() + offset, &value, 4);
}

auto nifti_put_i16(tf::buffer<char> &bytes, std::size_t offset,
                   std::int16_t value) -> void {
  std::memcpy(bytes.begin() + offset, &value, 2);
}

} // namespace

TEST_CASE("nifti: a hostile gzip size field refuses cheaply", "[io][nifti]") {
  tf::buffer<char> hostile;
  hostile.allocate(19);
  std::memset(hostile.begin(), 0, 19);
  const unsigned char head[10] = {0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 255};
  std::memcpy(hostile.begin(), head, 10);
  const std::uint32_t isize = 0xffffffffu;
  std::memcpy(hostile.begin() + 15, &isize, 4);
  const std::size_t hostile_payload = 19 - 10 - 8;
  REQUIRE(std::uint64_t(isize) >
          std::uint64_t(hostile_payload) * tf::io::gz::max_expansion);
  REQUIRE(tf::io::gzip_inflated(nifti_bytes_range(hostile)).size() == 0);
  REQUIRE(tf::read_nifti<float>(nifti_bytes_range(hostile)).status ==
          tf::nifti_status::truncated);
}

TEST_CASE("nifti: a legal extreme ratio stays inside the gzip bound",
          "[io][nifti]") {
  tf::buffer<char> zeros;
  zeros.allocate(1 << 20);
  std::memset(zeros.begin(), 0, zeros.size());
  const auto gz = tf::io::gzip_deflated(nifti_bytes_range(zeros));
  REQUIRE(gz.size() > 18);
  const auto payload = gz.size() - 10 - 8;
  REQUIRE(std::uint64_t(zeros.size()) <=
          std::uint64_t(payload) * tf::io::gz::max_expansion);
  REQUIRE(zeros.size() > 500 * payload);
  const auto inflated = tf::io::gzip_inflated(nifti_bytes_range(gz));
  REQUIRE(inflated.size() == zeros.size());
  REQUIRE(std::memcmp(inflated.begin(), zeros.begin(), zeros.size()) == 0);
}

TEST_CASE("nifti: an absurd vox_offset refuses before any conversion",
          "[io][nifti]") {
  const auto volume = nifti_test_volume<std::int16_t>();
  auto bytes = nifti_copy(tf::write_nifti_to_buffer(volume));
  nifti_put_f32(bytes, 108, 1e20f);
  REQUIRE(tf::read_nifti<float>(nifti_bytes_range(bytes)).status ==
          tf::nifti_status::truncated);
  nifti_put_f32(bytes, 108, 350.f);
  REQUIRE(tf::read_nifti<float>(nifti_bytes_range(bytes)).status ==
          tf::nifti_status::truncated);
}

TEST_CASE("nifti: an extent past the format refuses the write",
          "[io][nifti]") {
  tf::volume_buffer<std::uint8_t, float, 3> wide({40000, 1, 1},
                                                 {1.f, 1.f, 1.f},
                                                 {0.f, 0.f, 0.f});
  std::memset(wide.samples_buffer().begin(), 0, wide.voxel_count());
  REQUIRE(tf::write_nifti_to_buffer(wide).size() == 0);
  const auto path = nifti_temp_path(".nii");
  nifti_temp_file_cleanup cleanup{path};
  REQUIRE_FALSE(tf::write_nifti(wide, path.string()));
  REQUIRE_FALSE(std::filesystem::exists(path));
}

TEST_CASE("nifti: the qform quaternion answers the spec formula",
          "[io][nifti]") {
  const auto volume = nifti_test_volume<std::int16_t>();
  for (const auto qfac : {1.f, -1.f}) {
    auto bytes = nifti_copy(tf::write_nifti_to_buffer(volume));
    nifti_put_i16(bytes, 254, 0);
    nifti_put_i16(bytes, 252, 1);
    const float b = 0.f, c = 0.f, d = 0.70710678f;
    const float a = std::sqrt(std::max(0.f, 1.f - b * b - c * c - d * d));
    nifti_put_f32(bytes, 256, b);
    nifti_put_f32(bytes, 260, c);
    nifti_put_f32(bytes, 264, d);
    nifti_put_f32(bytes, 268, 4.f);
    nifti_put_f32(bytes, 272, 5.f);
    nifti_put_f32(bytes, 276, 6.f);
    nifti_put_f32(bytes, 76, qfac);
    const auto file = tf::read_nifti<float>(nifti_bytes_range(bytes));
    REQUIRE(bool(file));
    REQUIRE(file.posed);
    REQUIRE(file.reflecting == (qfac < 0.f));
    const float expected[3][3] = {
        {a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d,
         (2 * b * d + 2 * a * c) * qfac},
        {2 * b * c + 2 * a * d, a * a + c * c - b * b - d * d,
         (2 * c * d - 2 * a * b) * qfac},
        {2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b,
         (a * a + d * d - b * b - c * c) * qfac}};
    for (std::size_t r = 0; r < 3; ++r)
      for (std::size_t col = 0; col < 3; ++col)
        REQUIRE(std::abs(file.frame.transformation()(r, col) -
                         expected[r][col]) < 1e-5f);
    REQUIRE(file.frame.transformation()(0, 3) == 4.f);
    REQUIRE(file.frame.transformation()(1, 3) == 5.f);
    REQUIRE(file.frame.transformation()(2, 3) == 6.f);
    for (std::size_t axis = 0; axis < 3; ++axis)
      REQUIRE(file.volume.spacing()[axis] == volume.spacing()[axis]);
  }
}

TEST_CASE("nifti: a sheared sform recomposes to the same physical map",
          "[io][nifti]") {
  const auto volume = nifti_test_volume<float>();
  auto bytes = nifti_copy(tf::write_nifti_to_buffer(volume));
  const float srow[12] = {2.f, 0.5f, 0.f, 1.f, 0.f, 2.f,
                          0.f, 2.f,  0.f, 0.f, 3.f, 3.f};
  for (std::size_t i = 0; i < 12; ++i)
    nifti_put_f32(bytes, 280 + 4 * i, srow[i]);
  const auto file = tf::read_nifti<float>(nifti_bytes_range(bytes));
  REQUIRE(bool(file));
  REQUIRE(file.posed);
  const auto physical = [&](int i, int j, int k, std::size_t r) {
    return srow[4 * r] * float(i) + srow[4 * r + 1] * float(j) +
           srow[4 * r + 2] * float(k) + srow[4 * r + 3];
  };
  for (const auto &corner :
       {std::array<int, 3>{0, 0, 0}, std::array<int, 3>{4, 3, 2}}) {
    const auto read = tf::transformed(
        file.volume.point_at(corner[0], corner[1], corner[2]), file.frame);
    for (std::size_t r = 0; r < 3; ++r)
      REQUIRE(std::abs(read[r] -
                       physical(corner[0], corner[1], corner[2], r)) < 1e-4f);
  }
  const auto rewritten = tf::write_nifti_to_buffer(file.volume, file.frame);
  const auto again = tf::read_nifti<float>(nifti_bytes_range(rewritten));
  REQUIRE(bool(again));
  for (const auto &corner :
       {std::array<int, 3>{0, 0, 0}, std::array<int, 3>{4, 3, 2}}) {
    const auto read = tf::transformed(
        again.volume.point_at(corner[0], corner[1], corner[2]), again.frame);
    for (std::size_t r = 0; r < 3; ++r)
      REQUIRE(std::abs(read[r] -
                       physical(corner[0], corner[1], corner[2], r)) < 1e-3f);
  }
}

TEST_CASE("nifti: an LAS sform poses and reflects", "[io][nifti]") {
  const auto volume = nifti_test_volume<std::int16_t>();
  auto bytes = nifti_copy(tf::write_nifti_to_buffer(volume));
  const float srow[12] = {-1.f, 0.f, 0.f, 4.f, 0.f, 1.f,
                          0.f,  5.f, 0.f, 0.f, 1.f, 6.f};
  for (std::size_t i = 0; i < 12; ++i)
    nifti_put_f32(bytes, 280 + 4 * i, srow[i]);
  const auto file = tf::read_nifti<std::int16_t>(nifti_bytes_range(bytes));
  REQUIRE(bool(file));
  REQUIRE(file.posed);
  REQUIRE(file.reflecting);
  const auto info = tf::read_nifti_header(nifti_bytes_range(bytes));
  REQUIRE(info.posed);
  REQUIRE(info.reflecting);
}

TEST_CASE("nifti: pixdim alone places an unposed grid", "[io][nifti]") {
  const auto volume = nifti_test_volume<std::int16_t>();
  auto bytes = nifti_copy(tf::write_nifti_to_buffer(volume));
  nifti_put_i16(bytes, 254, 0);
  nifti_put_i16(bytes, 252, 0);
  const auto file = tf::read_nifti<std::int16_t>(nifti_bytes_range(bytes));
  REQUIRE(bool(file));
  REQUIRE_FALSE(file.posed);
  REQUIRE_FALSE(file.reflecting);
  for (std::size_t axis = 0; axis < 3; ++axis) {
    REQUIRE(file.volume.spacing()[axis] == volume.spacing()[axis]);
    REQUIRE(file.volume.origin()[axis] == 0.f);
  }
}

TEST_CASE("nifti: a zero slope leaves samples unscaled", "[io][nifti]") {
  const auto volume = nifti_test_volume<std::int16_t>();
  auto bytes = nifti_copy(tf::write_nifti_to_buffer(volume));
  nifti_put_f32(bytes, 112, 0.f);
  nifti_put_f32(bytes, 116, 5.f);
  const auto file = tf::read_nifti<std::int16_t>(nifti_bytes_range(bytes));
  REQUIRE(bool(file));
  REQUIRE(nifti_samples_equal(volume, file.volume));
  const auto info = tf::read_nifti_header(nifti_bytes_range(bytes));
  REQUIRE(info.slope == 0.f);
  REQUIRE(info.intercept == 5.f);
  REQUIRE(info.units == tf::nifti_units::millimeters);
}

TEST_CASE("nifti: corrupt or doubled gzip refuses as truncated",
          "[io][nifti]") {
  const auto volume = nifti_test_volume<std::int16_t>();
  const auto bytes = tf::write_nifti_to_buffer(volume);
  auto gz = nifti_copy(tf::io::gzip_deflated(nifti_bytes_range(bytes)));
  SECTION("a flipped payload byte") {
    gz.begin()[gz.size() / 2] ^= char(0x40);
    REQUIRE(tf::read_nifti<std::int16_t>(nifti_bytes_range(gz)).status ==
            tf::nifti_status::truncated);
  }
  SECTION("a false trailer crc") {
    gz.begin()[gz.size() - 8] ^= char(0x40);
    REQUIRE(tf::read_nifti<std::int16_t>(nifti_bytes_range(gz)).status ==
            tf::nifti_status::truncated);
  }
  SECTION("two members") {
    tf::buffer<char> doubled;
    doubled.allocate(2 * gz.size());
    std::memcpy(doubled.begin(), gz.begin(), gz.size());
    std::memcpy(doubled.begin() + gz.size(), gz.begin(), gz.size());
    REQUIRE(tf::io::gzip_inflated(nifti_bytes_range(doubled)).size() == 0);
    REQUIRE(tf::read_nifti<std::int16_t>(nifti_bytes_range(doubled)).status ==
            tf::nifti_status::truncated);
  }
}

TEST_CASE("nifti: serial and parallel gzip crc agree", "[io][nifti]") {
  constexpr std::size_t size = (5U << 20) + 17;
  tf::buffer<char> bytes;
  bytes.allocate(size + 8);
  std::uint32_t lcg = 7;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    lcg = lcg * 1664525u + 1013904223u;
    bytes.begin()[i] = char(lcg >> 24);
  }
  for (const auto offset : {std::size_t{0}, std::size_t{1}, std::size_t{7}})
    for (const auto length : {std::size_t{0}, std::size_t{1}, std::size_t{4},
                              std::size_t{7}, std::size_t{8},
                              (std::size_t{1} << 20) - 1,
                              std::size_t{1U << 20}, size}) {
      const auto *begin = bytes.begin() + offset;
      const auto reference = static_cast<std::uint32_t>(mz_crc32(
          MZ_CRC32_INIT, reinterpret_cast<const unsigned char *>(begin),
          length));
      REQUIRE(tf::io::gz::gzip_crc32(begin, length) == reference);
    }
}

TEST_CASE("nifti: the gzip prefix survives the window wrap", "[io][nifti]") {
  tf::buffer<char> pattern;
  pattern.allocate(100000);
  std::uint32_t lcg = 7;
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    lcg = lcg * 1664525u + 1013904223u;
    pattern.begin()[i] = char(lcg >> 24);
  }
  const auto gz = tf::io::gzip_deflated(nifti_bytes_range(pattern));
  const auto prefix =
      tf::io::gzip_inflated_prefix(nifti_bytes_range(gz), 50000);
  REQUIRE(prefix.size() == 50000);
  REQUIRE(std::memcmp(prefix.begin(), pattern.begin(), 50000) == 0);
}

TEST_CASE("nifti: dim zero and dim eight refuse, bitpix must agree",
          "[io][nifti]") {
  const auto volume = nifti_test_volume<std::int16_t>();
  SECTION("dim[0] = 0") {
    auto bytes = nifti_copy(tf::write_nifti_to_buffer(volume));
    nifti_put_i16(bytes, 40, 0);
    REQUIRE(tf::read_nifti<float>(nifti_bytes_range(bytes)).status ==
            tf::nifti_status::unsupported_dims);
  }
  SECTION("dim[0] = 8") {
    auto bytes = nifti_copy(tf::write_nifti_to_buffer(volume));
    nifti_put_i16(bytes, 40, 8);
    REQUIRE(tf::read_nifti<float>(nifti_bytes_range(bytes)).status ==
            tf::nifti_status::unsupported_dims);
  }
  SECTION("a contradictory bitpix refuses, a zero one is tolerated") {
    auto bytes = nifti_copy(tf::write_nifti_to_buffer(volume));
    nifti_put_i16(bytes, 72, 32);
    REQUIRE(tf::read_nifti<float>(nifti_bytes_range(bytes)).status ==
            tf::nifti_status::inconsistent_header);
    nifti_put_i16(bytes, 72, 0);
    const auto file = tf::read_nifti<std::int16_t>(nifti_bytes_range(bytes));
    REQUIRE(bool(file));
    REQUIRE(nifti_samples_equal(volume, file.volume));
  }
}
