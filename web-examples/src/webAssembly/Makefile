SRC_DIR := src
DIST_DIR := dist
SRC := $(SRC_DIR)/native.cpp
OUT_JS := $(DIST_DIR)/native.js

build:
	mkdir -p $(DIST_DIR)
	em++ $(SRC) -O3 \
	    -I$(SRC_DIR) \
		-s WASM=1 \
		-s MODULARIZE=1 \
		-s EXPORT_ES6=1 \
		-s ENVIRONMENT=web \
		-s ALLOW_MEMORY_GROWTH=1 \
		-lembind \
		--emit-tsd native.d.ts \
		-o $(OUT_JS)