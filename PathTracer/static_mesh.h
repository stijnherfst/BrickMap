#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
struct StaticMesh {

public:
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> normals;
	std::vector<glm::uvec3> faces;
	//TODO(Dan): Remove indices since it's deprecated, prefer faces vector
	std::vector<unsigned int> indices;

	StaticMesh() = default;

	int load(const aiScene*);
};