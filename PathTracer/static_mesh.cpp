#include "stdafx.h"

int StaticMesh::load(const aiScene* scene) {

	if (scene) {
		const aiMesh* mesh = scene->mMeshes[0];
		vertices.reserve(mesh->mNumVertices);
		normals.reserve(mesh->mNumVertices);
		indices.reserve(mesh->mNumFaces * 3);
		faces.reserve(mesh->mNumFaces);

		for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
			const aiVector3D* vertex = &(mesh->mVertices[i]);
			const aiVector3D* normal = &(mesh->mNormals[i]);

			// Switch y and z
			vertices.push_back(glm::vec3(vertex->x, vertex->z, vertex->y));
			normals.push_back(glm::vec3(normal->x, normal->y, normal->z));
		}

		for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
			const aiFace& face = mesh->mFaces[i];
			indices.push_back(face.mIndices[0]);
			indices.push_back(face.mIndices[1]);
			indices.push_back(face.mIndices[2]);

			faces.push_back(glm::uvec3(face.mIndices[0], face.mIndices[1], face.mIndices[2]));
		}
	}

	return 0;
}