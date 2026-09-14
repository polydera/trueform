if(NOT DEFINED LINK_MAP OR LINK_MAP STREQUAL "")
  message(FATAL_ERROR "LINK_MAP is required")
endif()
if(NOT EXISTS "${LINK_MAP}")
  message(FATAL_ERROR "float 2D ray-cast link map does not exist: ${LINK_MAP}")
endif()

file(READ "${LINK_MAP}" link_map)

string(FIND "${link_map}" "ray_cast_float_2d.cpp.o" float_2d_position)
if(float_2d_position EQUAL -1)
  message(FATAL_ERROR
    "float 2D ray-cast link omitted its explicit-instantiation shard")
endif()

string(REPLACE "ray_cast_float_2d.cpp.o" "" unrelated_link_map "${link_map}")
if(unrelated_link_map MATCHES
    "intersects_(float|double|mixed)(|_2d)\\.cpp\\.o|neighbor_search_(float|double)\\.cpp\\.o|ray_cast_(float|double|mixed)(|_2d)\\.cpp\\.o|gather_ids_(float|double)_(2d|3d)\\.cpp\\.o")
  message(FATAL_ERROR
    "float/int64/2D ray-cast link extracted an unrelated spatial operation shard")
endif()

message(STATUS
  "Float/int64/2D ray-cast link extracted only its requested operation shard")
