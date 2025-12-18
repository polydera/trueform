FROM emscripten/emsdk:4.0.15 AS web-builder

# Build dependencies
RUN apt-get update && apt-get install -y ninja-build git cmake && rm -rf /var/lib/apt/lists/*

# Install TBB library as a system-wide library
ARG TBB_VERSION=2021.13.1
ENV TBB_VERSION=${TBB_VERSION}

WORKDIR /deps/tbb
RUN git clone --depth 1 --branch v${TBB_VERSION} https://github.com/oneapi-src/oneTBB.git .
RUN mkdir build && cd build && \
    emcmake cmake ..  \
    -DCMAKE_CXX_FLAGS="-fwasm-exceptions" \
    -DBUILD_SHARED_LIBS=OFF  \
    -DTBB_TEST=OFF  \
    -DTBB_ENABLE_IPO=OFF \
    -DTBBMALLOC_BUILD=OFF \
    -DTBBMALLOC_PROXY_BUILD=OFF \
    -DTBB_DISABLE_HWLOC_AUTOMATIC_SEARCH=ON \
    -DCMAKE_BUILD_TYPE=Release  \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_POLICY_VERSION_MINIMUM="3.4" \
    -G Ninja && \
    ninja && \
    ninja install

RUN cd /emsdk/upstream/emscripten && npm install --no-save --package-lock false typescript

# Build the LTO enabled example so that we cache the system libs
WORKDIR /src/wasm-examples/extras/lto
RUN echo "Building LTO example to cache system libs..." && \
    echo "cmake_minimum_required(VERSION 3.13)" > CMakeLists.txt && \
    echo "project(lto_example)" >> CMakeLists.txt && \
    echo "add_compile_options(-flto -fwasm-exceptions -pthread -msimd128 -O3)" >> CMakeLists.txt && \
    echo "add_link_options(-O3 -flto -fwasm-exceptions -pthread)" >> CMakeLists.txt && \
    echo "add_executable(lto_example lto_example.cpp)" >> CMakeLists.txt && \
    echo "int main() { return 0; }" > lto_example.cpp && \
    emcmake cmake . -DCMAKE_BUILD_TYPE="Release" -DCMAKE_CXX_FLAGS="-flto -fwasm-exceptions" -DCMAKE_EXE_LINKER_FLAGS="-flto" && \
    emmake make -j$(nproc)