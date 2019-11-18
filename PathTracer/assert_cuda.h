#pragma once

cudaError cuda_assert(const cudaError code, const char* const file, const int line, const bool abort);

#define cuda(...) cuda_assert((cuda##__VA_ARGS__), __FILE__, __LINE__, true);