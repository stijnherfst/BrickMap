#include "stdafx.h"

cudaError cuda_assert(const cudaError code, const char* const file, const int line, const bool abort) {
	if (code != cudaSuccess) {
		fprintf(stderr, "cuda_assert: %s %s %d\n", cudaGetErrorString(code), file, line);

		if (abort) {
			cudaDeviceReset();
			exit(code);
		}
	}

	return code;
}