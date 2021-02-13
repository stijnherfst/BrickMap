#pragma once

#ifndef __CUDACC__
#define __launch_bounds__()

int atomicAdd(void*, unsigned int){};
int atomicAnd(int* address, int val){};
unsigned int atomicAnd(unsigned int* address, unsigned int val) {}
unsigned int atomicOr(unsigned int* address, unsigned int val) {}

template <typename T, int TT>
class surface {
};
template <typename T, int TT>
class texture {
};
#endif