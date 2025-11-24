SHELL := /bin/bash

BUILD_DIR ?= build
EMSDK_ENV := source emsdk/emsdk_env.sh >/dev/null &&

.PHONY: all configure build clean

all: build

configure:
	$(EMSDK_ENV) emcmake cmake -S . -B $(BUILD_DIR)

build: configure
	$(EMSDK_ENV) cmake --build $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
