#pragma once

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;

	Vertex() = default;
	Vertex(float x, float y, float z, float nx, float ny, float nz)
		: position(x, y, z)
		, normal(glm::vec3(nx, ny, nz)) {}
};

struct Triangle {
	glm::vec3 vert;
	glm::vec3 e1, e2;
	uint8_t materialType{};
	// HACK to align to 64 bytes since __align__ doesn't seem to work
	// char placeholder_align[64 - 4 * sizeof(float3) - sizeof(uint8_t)];

	//Muller-Trumbore intersection
	__device__ float intersect(const glm::vec3& origin, const glm::vec3& direction) {
		float u, v;
		glm::vec3 pvec = glm::cross(direction, e2);
		float det = glm::dot(e1, pvec);

		// if the determinant is negative the triangle is backfacing
		// if the determinant is close to 0, the ray misses the triangle
		if (det < 0.0000001f)
			return false;
		// ray and triangle are parallel if det is close to 0
		// if (fabs(det) < Epsilon) return 0;
		float invDet = 1 / det;

		glm::vec3 tvec = origin - vert;
		u = glm::dot(tvec, pvec) * invDet;
		if (u < 0 || u > 1)
			return 0;

		glm::vec3 qvec = glm::cross(tvec, e1);
		v = glm::dot(direction, qvec) * invDet;
		if (v < 0 || u + v > 1)
			return 0;
		float t = dot(e2, qvec) * invDet;

		return t;
	}
};
