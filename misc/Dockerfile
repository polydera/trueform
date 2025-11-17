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
