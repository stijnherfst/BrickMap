if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
	set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif ()
if (DEFINED ENV{VCPKG_DEFAULT_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
	set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}" CACHE STRING "")
endif ()

cmake_minimum_required(VERSION 3.18)
project(Brickmap LANGUAGES CUDA CXX)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 14)
# set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -Wundef")
# set(CMAKE_CXX_FLAGS_DEBUG "-march=native -g")
# set(CMAKE_CXX_FLAGS_RELEASE "-march=native -g -O3 -ffast-math")

# # Ignore some warnings that add a lot of spam and aren't that critical:
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-parameter")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-parentheses")

find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(soil2 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 61)

add_executable(Brickmap
	"src/assert_cuda.cpp"
	"src/camera.cpp"
	"src/imgui.ini"
	"src/imgui_impl_glfw.cpp"
	"src/imgui_impl_opengl3.cpp"
	"src/interop.cpp"
	"src/kernel.cu"
	"src/main.cpp"
	"src/performance_measure.cpp"
	"src/Scene.cpp"
	"src/SimplexNoise.cpp"
	"src/stdafx.cpp"
	"src/sunsky.cu"
	"src/sunsky.cuh"
	"src/variables.cpp"
	"src/voxel.cu"
	"src/voxel.cuh"

	"src/FastNoiseSIMD/FastNoiseSIMD.cpp"
	"src/FastNoiseSIMD/FastNoiseSIMD.h"
	"src/FastNoiseSIMD/FastNoiseSIMD_avx2.cpp"
	"src/FastNoiseSIMD/FastNoiseSIMD_avx512.cpp"
	"src/FastNoiseSIMD/FastNoiseSIMD_internal.cpp"
	"src/FastNoiseSIMD/FastNoiseSIMD_internal.h"
	"src/FastNoiseSIMD/FastNoiseSIMD_neon.cpp"
	"src/FastNoiseSIMD/FastNoiseSIMD_sse2.cpp"
	"src/FastNoiseSIMD/FastNoiseSIMD_sse41.cpp"
)

target_link_libraries(Brickmap PRIVATE
	glad::glad
	glfw
	glm
	soil2
	imgui::imgui
)

target_include_directories(Brickmap PRIVATE 
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
set_target_properties(Brickmap PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

target_compile_options(Brickmap PRIVATE                        
	$<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
)

target_compile_features(Brickmap PUBLIC cxx_std_20)