#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <unordered_map>

#include "glad/glad.h"

#include "GLFW/glfw3.h"

#include "cuda.h"
#include <cuda_runtime.h>

#include "glm/glm.hpp"
#include "glm/gtc/integer.hpp" 

#include "FastNoiseSIMD/FastNoiseSIMD.h"
#include "SimplexNoise.h"
#include "variables.h"
#include "assert_cuda.h"
#include "cuda_gl_interop.h"
#include "performance_measure.h"

#include "Scene.h"

#include "camera.h"

#include "interop.h"