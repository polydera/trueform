macro(GET_TBB)
    if(USE_SYSTEM_TBB AND BUILD_PYTHON)
        message(FATAL_ERROR "Python bindings require static TBB which is incompatible with system TBB.")
    endif()

    if (USE_SYSTEM_TBB)
        find_pacakge(TBB REQUIRED)
    else ()
        include(FetchContent)
        FetchContent_Declare(
                tbb
                GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
                GIT_TAG v2021.13.1)
        set(TBB_BUILD
                ON
                CACHE BOOL "Enable TBB" FORCE)
        set(TBBMALLOC_BUILD
                OFF
                CACHE BOOL "Disable TBB malloc" FORCE)
        set(TBBMALLOC_PROXY_BUILD
                OFF
                CACHE BOOL "Disable TBB proxy" FORCE)
        set(TBBBIND_BUILD
                OFF
                CACHE BOOL "Disable TBB bind" FORCE)
        set(TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH
                ON
                CACHE BOOL "Disable hwloc automatic search" FORCE)
        set(TBB_INSTALL
                ON
                CACHE BOOL "Enable TBB install" FORCE)
        set(TBB_TEST
                OFF
                CACHE BOOL "Disable TBB tests" FORCE)
        set(TMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
        if(BUILD_PYTHON)
            message(STATUS "TBB: Forcing static build due to Python bindings requirement (BUILD_PYTHON=ON).")
            set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build type" FORCE)
        endif()
        FetchContent_MakeAvailable(tbb)
        set(BUILD_SHARED_LIBS ${TMP_BUILD_SHARED_LIBS} CACHE BOOL "Build type" FORCE)
    endif ()
endmacro()

GET_TBB()