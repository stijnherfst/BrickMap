#pragma once

struct BBox {
	//bounds[0] = bottom, bounds[1] = top
	glm::vec3 bounds[2] = { { 1e10, 1e10, 1e10 }, { -1e10, -1e10, -1e10 } };

	/*Add a vertex to the encompassing bounding box, enlarging it as needed*/
	__host__ BBox& addVertex(const glm::vec3& vertex) {
		bounds[0] = glm::vec3(fmin(bounds[0].x, vertex.x), fmin(bounds[0].y, vertex.y),
							  fmin(bounds[0].z, vertex.z));
		bounds[1] = glm::vec3(fmax(bounds[1].x, vertex.x), fmax(bounds[1].y, vertex.y),
							  fmax(bounds[1].z, vertex.z));
		return *this;
	}

	//Computes the vector top - botom (i.e diagonal of the box)
	__host__ glm::vec3 diagonal() const { return bounds[1] - bounds[0]; }
	__host__ float surfaceArea() const {
		glm::vec3 d = diagonal();
		return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
	}
	__host__ float volume() const {
		glm::vec3 d = diagonal();
		return d.x * d.y * d.z;
	}

	//Calculates which dimension is largest for the bbox 
	__host__ int largestExtent() const {
		glm::vec3 d = diagonal();
		if (d.x > d.y && d.x > d.z)
			return 0; // X is largest
		else if (d.y > d.z)
			return 1; // Y is largest
		else
			return 2; // Z is largest
	}

	__device__ bool intersect(const glm::vec3& origin, glm::vec3 invDir, int rayDirNeg[3], float lowestIntersect) const {
		float tMin = (bounds[rayDirNeg[0]].x - origin.x) * invDir.x;
		float tMax = (bounds[1 - rayDirNeg[0]].x - origin.x) * invDir.x;
		float tyMin = (bounds[rayDirNeg[1]].y - origin.y) * invDir.y;
		float tyMax = (bounds[1 - rayDirNeg[1]].y - origin.y) * invDir.y;

		if (tMin > tyMax || tyMin > tMax)
			return false;
		if (tyMin > tMin)
			tMin = tyMin;
		if (tyMax < tMax)
			tMax = tyMax;

		float tzMin = (bounds[rayDirNeg[2]].z - origin.z) * invDir.z;
		float tzMax = (bounds[1 - rayDirNeg[2]].z - origin.z) * invDir.z;

		if (tMin > tzMax || tzMin > tMax)
			return false;
		if (tzMin > tMin)
			tMin = tzMin;
		if (tzMax < tMax)
			tMax = tzMax;

		return (tMin < lowestIntersect) && (tMax > 0);
	}
};

/*Computes the union of two bounding boxes*/
__host__ BBox Union(const BBox& b1, const BBox& b2);
