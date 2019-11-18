#include "stdafx.h"

__host__ BBox Union(const BBox& b1, const BBox& b2) {
	BBox newBBox;
	newBBox.bounds[0].x = fmin(b1.bounds[0].x, b2.bounds[0].x);
	newBBox.bounds[0].y = fmin(b1.bounds[0].y, b2.bounds[0].y);
	newBBox.bounds[0].z = fmin(b1.bounds[0].z, b2.bounds[0].z);

	newBBox.bounds[1].x = fmax(b1.bounds[1].x, b2.bounds[1].x);
	newBBox.bounds[1].y = fmax(b1.bounds[1].y, b2.bounds[1].y);
	newBBox.bounds[1].z = fmax(b1.bounds[1].z, b2.bounds[1].z);

	return newBBox;
}
