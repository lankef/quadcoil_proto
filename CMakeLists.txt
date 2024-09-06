cmake_minimum_required(VERSION 3.14)
project(biest_call LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and NumPy
set(Python_FIND_STRATEGY LOCATION)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

# Find pybind11 using Python
execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})

# Add xtensor and xtensor-python as subdirectories because they
# have their own CMakeLists.txt
add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/xtl/)
add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/xtensor/)
add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/xtensor-python/)

# Adding BIEST and SCTL as part of the current CMakeLists.txt
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/SCTL/include/)
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/BIEST/include/)

# Find packages
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

# Find optional packages
find_package(BLAS REQUIRED)
find_package(LAPACK)
# Not available on mac.
# find_library(MVEC_LIB mvec)
# Has missing /openmpi-4.1.2-pkgs/fftw-3.3.10/lib/cmake/fftw3/FFTW3LibraryDepends.cmake 
# on the pppl cluster
# find_package(FFTW3)

# fastmath
# set(ENABLE_FAST_MATH ON)

# Compiler flags
# Note to self: the compiler flag must be set AFTER finding OpenMP.
set(COMMON_FLAGS "${COMMON_FLAGS} -fopenmp")
# set(COMMON_FLAGS "${COMMON_FLAGS} -DSCTL_GLOBAL_MEM_BUFF=0")
# set(COMMON_FLAGS "-Wall -Wfloat-conversion") # Optional: Release flags
set(OPTIONAL_LIB_FLAGS "-lblas -DSCTL_HAVE_BLAS -llapack -DSCTL_HAVE_LAPACK")
# FFTW cannot be found reliably using find_package on mac or the PPPL cluster.
# We'll fix this later.
# set(MVEC_FLAGS "-DSCTL_HAVE_LIBMVEC")
# set(FFTW_FLAGS "-lfftw3_omp -DSCTL_FFTW_THREADS -lfftw3 -DSCTL_HAVE_FFTW -lfftw3f -DSCTL_HAVE_FFTWF -lfftw3l -DSCTL_HAVE_FFTWL")
# Optional: Debug flags
set(DEBUG_FLAGS "-O0 -fsanitize=address,leak,undefined,pointer-compare,pointer-subtract,float-divide-by-zero,float-cast-overflow -fno-sanitize-recover=all -fstack-protector -DSCTL_MEMDEBUG")
set(RELEASE_FLAGS "-O3 -DNDEBUG")
# set(RELEASE_FLAGS "-O3 -march=native -DNDEBUG")
# Setting flags
set(CMAKE_CXX_FLAGS "${COMMON_FLAGS} ${RELEASE_FLAGS} ${PLATFORM_FLAGS} ${OPTIONAL_LIB_FLAGS}")


# Define your extension
pybind11_add_module(biest_call src/biest_call_cpp/biest_call.cpp)

target_link_libraries(biest_call PRIVATE 
    OpenMP::OpenMP_CXX 
    xtensor 
    xtensor-python 
    pybind11::module
    blas
    lapack 
)
# Not available on mac
# target_link_libraries(biest_call PRIVATE ${MVEC_LIB})
# Add back later when we fix it
# target_link_libraries(biest_call PRIVATE 
#     fftw3 
#     fftw3_omp 
#     fftw3f 
#     fftw3l 
# )
target_include_directories(biest_call PRIVATE 
    ${Python_INCLUDE_DIRS} 
    ${Python_NumPy_INCLUDE_DIRS}
)
