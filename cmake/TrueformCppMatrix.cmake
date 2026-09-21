# The instantiation matrix of the C++ facade.
#
# TF_CPP_REALS / TF_CPP_INDICES / TF_CPP_DIMS / TF_CPP_LAYOUTS say which
# coordinate types, index widths, dimensions and face layouts the archive
# carries. The first three select shard translation units by the tokens in
# their file names -- a shard is kept only when every token it names is
# enabled, so a two-index or mixed-precision shard needs all of the types it
# serves. All four generate one header stating the same matrix to the
# compiler; a shard is per (index, real, dims) and crosses the layout axis
# inside, exactly as a per-real shard crosses the index axis, so the layout is
# an instantiation filter rather than a file-name token.

set(TF_CPP_MATRIX_ALL_REALS float double)
set(TF_CPP_MATRIX_ALL_INDICES int32 int64)
set(TF_CPP_MATRIX_ALL_DIMS 2 3)
set(TF_CPP_MATRIX_ALL_LAYOUTS triangles dynamic)

function(tf_cpp_require_matrix_values name allowed)
  set(values ${ARGN})
  if(NOT values)
    message(FATAL_ERROR "${name} must name at least one value; allowed: ${${allowed}}")
  endif()
  foreach(value IN LISTS values)
    if(NOT value IN_LIST ${allowed})
      message(FATAL_ERROR
        "${name} names an unknown value '${value}'; allowed: ${${allowed}}")
    endif()
  endforeach()
endfunction()

# The C++ spelling of an index token.
function(tf_cpp_index_type token out)
  if(token STREQUAL "int32")
    set(${out} "std::int32_t" PARENT_SCOPE)
  else()
    set(${out} "std::int64_t" PARENT_SCOPE)
  endif()
endfunction()

# TRIANGLES IS THE LAYER'S OWN ARITY. Every result the layer mints -- a
# boolean, a triangulation, a primitive, an STL, a remesh, a csg arrangement --
# is a triangle mesh by its operation's own definition, so the triangle carrier
# is always built and the layout option says whether the MIXED carrier is
# carried beside it.
function(tf_cpp_require_triangle_layout)
  if(NOT "triangles" IN_LIST TF_CPP_LAYOUTS)
    message(FATAL_ERROR
      "TF_CPP_LAYOUTS must carry 'triangles': it is the arity every result the "
      "layer mints states")
  endif()
endfunction()

# The C++ spelling of a layout token, which is a face arity.
function(tf_cpp_ngon_value token out)
  if(token STREQUAL "triangles")
    set(${out} "3" PARENT_SCOPE)
  else()
    set(${out} "tf::dynamic_size" PARENT_SCOPE)
  endif()
endfunction()

# Keep a shard only when every type its file name names is enabled. A name
# states the axes its shard is per, and NOTHING ELSE: the layer's own carriers
# -- nd_array, curves, index_map, mesh_structure, obb, cdt -- cross the whole
# matrix inside one translation unit and name no axis at all, so a token a name
# does not carry is an axis the shard does not have and never a default.
# index_map and offset_blocked_buffer state both widths regardless of the
# matrix: a point cloud's ids and default_index_t are int32 by construction,
# and as_offset_blocked reads int64, so neither width is the matrix's to drop.
function(tf_cpp_matrix_filter_sources out)
  set(kept "")
  foreach(source IN LISTS ARGN)
    get_filename_component(stem "${source}" NAME_WE)
    string(REPLACE "_" ";" tokens "${stem}")
    set(keep TRUE)
    foreach(token IN LISTS tokens)
      if(token STREQUAL "mixed")
        # A mixed shard is one entry read at two precisions, so it needs both.
        foreach(real IN LISTS TF_CPP_MATRIX_ALL_REALS)
          if(NOT real IN_LIST TF_CPP_REALS)
            set(keep FALSE)
          endif()
        endforeach()
      elseif(token IN_LIST TF_CPP_MATRIX_ALL_REALS)
        if(NOT token IN_LIST TF_CPP_REALS)
          set(keep FALSE)
        endif()
      elseif(token IN_LIST TF_CPP_MATRIX_ALL_INDICES)
        if(NOT token IN_LIST TF_CPP_INDICES)
          set(keep FALSE)
        endif()
      elseif(token STREQUAL "2d" OR token STREQUAL "3d")
        string(SUBSTRING "${token}" 0 1 dimension)
        if(NOT dimension IN_LIST TF_CPP_DIMS)
          set(keep FALSE)
        endif()
      endif()
    endforeach()
    if(keep)
      list(APPEND kept "${source}")
    endif()
  endforeach()
  set(${out} "${kept}" PARENT_SCOPE)
endfunction()

# Add a module's shards, less the ones this matrix does not carry.
function(tf_cpp_add_shards)
  tf_cpp_matrix_filter_sources(kept ${ARGN})
  if(kept)
    target_sources(trueform_cpp PRIVATE ${kept})
  endif()
endfunction()

# Write the one header that states the matrix to the compiler: an iteration
# macro per combination shape, so an extern-template list is the matrix
# rather than a copy of it, and a trait per axis, so a combination outside it
# is refused at the carrier instead of at the link.
function(tf_cpp_generate_matrix_header input output)
  set(indices "")
  foreach(token IN LISTS TF_CPP_INDICES)
    tf_cpp_index_type("${token}" type)
    list(APPEND indices "${type}")
  endforeach()

  set(ngons "")
  foreach(token IN LISTS TF_CPP_LAYOUTS)
    tf_cpp_ngon_value("${token}" value)
    list(APPEND ngons "${value}")
  endforeach()

  set(knows_real "")
  foreach(real IN LISTS TF_CPP_MATRIX_ALL_REALS)
    string(APPEND knows_real
      "\ntemplate <> inline constexpr bool matrix_knows_real_v<${real}> = true;")
  endforeach()

  set(for_each_real "")
  set(has_real "")
  foreach(real IN LISTS TF_CPP_REALS)
    string(APPEND for_each_real " M(${real});")
    string(APPEND has_real
      "\ntemplate <> inline constexpr bool matrix_has_real_v<${real}> = true;")
  endforeach()

  set(for_each_index "")
  set(has_index "")
  foreach(index IN LISTS indices)
    string(APPEND for_each_index " M(${index});")
    string(APPEND has_index
      "\ntemplate <> inline constexpr bool matrix_has_index_v<${index}> = true;")
  endforeach()

  set(for_each_dims "")
  set(has_dims "")
  foreach(dims IN LISTS TF_CPP_DIMS)
    string(APPEND for_each_dims " M(${dims});")
    string(APPEND has_dims
      "\ntemplate <> inline constexpr bool matrix_has_dims_v<${dims}> = true;")
  endforeach()

  set(for_each_ngon "")
  set(has_ngon "")
  foreach(ngon IN LISTS ngons)
    string(APPEND for_each_ngon " M(${ngon});")
    string(APPEND has_ngon
      "\ntemplate <> inline constexpr bool matrix_has_ngon_v<${ngon}> = true;")
  endforeach()

  set(for_each_ngon_pair "")
  foreach(ngon IN LISTS ngons)
    foreach(other IN LISTS ngons)
      string(APPEND for_each_ngon_pair " M(${ngon}, ${other});")
    endforeach()
  endforeach()

  set(for_each_real_dims "")
  set(for_each_real_ngon "")
  foreach(real IN LISTS TF_CPP_REALS)
    foreach(dims IN LISTS TF_CPP_DIMS)
      string(APPEND for_each_real_dims " M(${real}, ${dims});")
    endforeach()
    foreach(ngon IN LISTS ngons)
      string(APPEND for_each_real_ngon " M(${real}, ${ngon});")
    endforeach()
  endforeach()

  set(for_each_real_pair "")
  set(for_each_real_pair_dims "")
  foreach(real IN LISTS TF_CPP_REALS)
    foreach(other IN LISTS TF_CPP_REALS)
      string(APPEND for_each_real_pair " M(${real}, ${other});")
      foreach(dims IN LISTS TF_CPP_DIMS)
        string(APPEND for_each_real_pair_dims " M(${real}, ${other}, ${dims});")
      endforeach()
    endforeach()
  endforeach()

  set(for_each_index_pair "")
  set(for_each_index_pair_ngon "")
  set(for_each_index_pair_ngon_pair "")
  foreach(index IN LISTS indices)
    foreach(other IN LISTS indices)
      string(APPEND for_each_index_pair " M(${index}, ${other});")
      foreach(ngon IN LISTS ngons)
        string(APPEND for_each_index_pair_ngon
          " M(${index}, ${other}, ${ngon});")
        foreach(other_ngon IN LISTS ngons)
          string(APPEND for_each_index_pair_ngon_pair
            " M(${index}, ${other}, ${ngon}, ${other_ngon});")
        endforeach()
      endforeach()
    endforeach()
  endforeach()

  set(for_each_index_ngon "")
  foreach(index IN LISTS indices)
    foreach(ngon IN LISTS ngons)
      string(APPEND for_each_index_ngon " M(${index}, ${ngon});")
    endforeach()
  endforeach()

  set(for_each_index_real "")
  set(for_each_index_real_dims "")
  set(for_each_index_real_index "")
  set(for_each_index_real_ngon "")
  set(for_each_index_real_ngon_pair "")
  set(for_each_index_real_dims_ngon "")
  set(for_each_index_real_index_ngon_pair "")
  foreach(index IN LISTS indices)
    foreach(real IN LISTS TF_CPP_REALS)
      string(APPEND for_each_index_real " M(${index}, ${real});")
      foreach(dims IN LISTS TF_CPP_DIMS)
        string(APPEND for_each_index_real_dims " M(${index}, ${real}, ${dims});")
        foreach(ngon IN LISTS ngons)
          string(APPEND for_each_index_real_dims_ngon
            " M(${index}, ${real}, ${dims}, ${ngon});")
        endforeach()
      endforeach()
      foreach(ngon IN LISTS ngons)
        string(APPEND for_each_index_real_ngon
          " M(${index}, ${real}, ${ngon});")
        foreach(other_ngon IN LISTS ngons)
          string(APPEND for_each_index_real_ngon_pair
            " M(${index}, ${real}, ${ngon}, ${other_ngon});")
        endforeach()
      endforeach()
      foreach(other IN LISTS indices)
        string(APPEND for_each_index_real_index
          " M(${index}, ${real}, ${other});")
        foreach(ngon IN LISTS ngons)
          foreach(other_ngon IN LISTS ngons)
            string(APPEND for_each_index_real_index_ngon_pair
              " M(${index}, ${real}, ${other}, ${ngon}, ${other_ngon});")
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()

  set(for_each_real_index_real_dims "")
  set(for_each_real_index_real_dims_ngon "")
  foreach(real IN LISTS TF_CPP_REALS)
    foreach(index IN LISTS indices)
      foreach(other_real IN LISTS TF_CPP_REALS)
        foreach(dims IN LISTS TF_CPP_DIMS)
          string(APPEND for_each_real_index_real_dims
            " M(${real}, ${index}, ${other_real}, ${dims});")
          foreach(ngon IN LISTS ngons)
            string(APPEND for_each_real_index_real_dims_ngon
              " M(${real}, ${index}, ${other_real}, ${dims}, ${ngon});")
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()

  set(for_each_index_real_dims_index_real "")
  set(for_each_index_real_dims_index_real_ngon_pair "")
  foreach(index IN LISTS indices)
    foreach(real IN LISTS TF_CPP_REALS)
      foreach(dims IN LISTS TF_CPP_DIMS)
        foreach(other IN LISTS indices)
          foreach(other_real IN LISTS TF_CPP_REALS)
            string(APPEND for_each_index_real_dims_index_real
              " M(${index}, ${real}, ${dims}, ${other}, ${other_real});")
            foreach(ngon IN LISTS ngons)
              foreach(other_ngon IN LISTS ngons)
                string(APPEND for_each_index_real_dims_index_real_ngon_pair
                  " M(${index}, ${real}, ${dims}, ${other}, ${other_real}, ${ngon}, ${other_ngon});")
              endforeach()
            endforeach()
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()

  foreach(dims IN LISTS TF_CPP_MATRIX_ALL_DIMS)
    if(dims IN_LIST TF_CPP_DIMS)
      set(TF_CPP_MATRIX_HAS_${dims}D 1)
    else()
      set(TF_CPP_MATRIX_HAS_${dims}D 0)
    endif()
  endforeach()

  foreach(token IN LISTS TF_CPP_MATRIX_ALL_INDICES)
    string(TOUPPER "${token}" upper)
    if(token IN_LIST TF_CPP_INDICES)
      set(TF_CPP_MATRIX_HAS_${upper} 1)
    else()
      set(TF_CPP_MATRIX_HAS_${upper} 0)
    endif()
  endforeach()

  foreach(real IN LISTS TF_CPP_MATRIX_ALL_REALS)
    string(TOUPPER "${real}" upper)
    if(real IN_LIST TF_CPP_REALS)
      set(TF_CPP_MATRIX_HAS_${upper} 1)
    else()
      set(TF_CPP_MATRIX_HAS_${upper} 0)
    endif()
  endforeach()

  foreach(layout IN LISTS TF_CPP_MATRIX_ALL_LAYOUTS)
    string(TOUPPER "${layout}" upper)
    if(layout IN_LIST TF_CPP_LAYOUTS)
      set(TF_CPP_MATRIX_HAS_${upper} 1)
    else()
      set(TF_CPP_MATRIX_HAS_${upper} 0)
    endif()
  endforeach()

  set(TF_CPP_MATRIX_FOR_EACH_REAL "${for_each_real}")
  set(TF_CPP_MATRIX_FOR_EACH_INDEX "${for_each_index}")
  set(TF_CPP_MATRIX_FOR_EACH_DIMS "${for_each_dims}")
  set(TF_CPP_MATRIX_FOR_EACH_NGON "${for_each_ngon}")
  set(TF_CPP_MATRIX_FOR_EACH_NGON_PAIR "${for_each_ngon_pair}")
  set(TF_CPP_MATRIX_FOR_EACH_REAL_DIMS "${for_each_real_dims}")
  set(TF_CPP_MATRIX_FOR_EACH_REAL_NGON "${for_each_real_ngon}")
  set(TF_CPP_MATRIX_FOR_EACH_REAL_PAIR "${for_each_real_pair}")
  set(TF_CPP_MATRIX_FOR_EACH_REAL_PAIR_DIMS "${for_each_real_pair_dims}")
  set(TF_CPP_MATRIX_FOR_EACH_REAL_INDEX_REAL_DIMS
    "${for_each_real_index_real_dims}")
  set(TF_CPP_MATRIX_FOR_EACH_REAL_INDEX_REAL_DIMS_NGON
    "${for_each_real_index_real_dims_ngon}")
  set(TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_INDEX_REAL
    "${for_each_index_real_dims_index_real}")
  set(TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_INDEX_REAL_NGON_PAIR
    "${for_each_index_real_dims_index_real_ngon_pair}")
  set(TF_CPP_MATRIX_FOR_EACH_INDEX_PAIR "${for_each_index_pair}")
  set(TF_CPP_MATRIX_FOR_EACH_INDEX_NGON "${for_each_index_ngon}")
  set(TF_CPP_MATRIX_FOR_EACH_INDEX_PAIR_NGON "${for_each_index_pair_ngon}")
  set(TF_CPP_MATRIX_FOR_EACH_INDEX_PAIR_NGON_PAIR
    "${for_each_index_pair_ngon_pair}")
  set(TF_CPP_MATRIX_FOR_EACH_INDEX_REAL "${for_each_index_real}")
  set(TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS "${for_each_index_real_dims}")
  set(TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_INDEX "${for_each_index_real_index}")
  set(TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON "${for_each_index_real_ngon}")
  set(TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON_PAIR
    "${for_each_index_real_ngon_pair}")
  set(TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON
    "${for_each_index_real_dims_ngon}")
  set(TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_INDEX_NGON_PAIR
    "${for_each_index_real_index_ngon_pair}")
  set(TF_CPP_MATRIX_KNOWS_REAL "${knows_real}")
  set(TF_CPP_MATRIX_HAS_REAL "${has_real}")
  set(TF_CPP_MATRIX_HAS_INDEX "${has_index}")
  set(TF_CPP_MATRIX_HAS_DIMS "${has_dims}")
  set(TF_CPP_MATRIX_HAS_NGON "${has_ngon}")
  configure_file("${input}" "${output}" @ONLY)
endfunction()
