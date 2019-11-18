#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <numeric>
#include <stdbool.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "glad/glad.h"

#include "GLFW/glfw3.h"

#include "cuda.h"
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
//#define GLM_FORCE_PURE
#define GLM_FORCE_CXX17
#include "glm.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "Rays.h"

#include "variables.h"
#include "assert_cuda.h"
#include "cuda_gl_interop.h"
#include "loader.h"
#include "performance_measure.h"
#include "static_mesh.h"

#include "Bbox.h"
#include "BVH.h"
#include "Scene.h"

#include "camera.h"

#include "interop.h"