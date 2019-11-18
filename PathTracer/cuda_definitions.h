#pragma once

#ifndef __CUDACC__
	#define __launch_bounds__()

	int atomicAdd(void*, unsigned int){};

	template <typename T, int TT>
	class surface {
	};
	template <typename T, int TT>
	class texture {
	};
#endif