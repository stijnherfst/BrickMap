#include "stdafx.h"
#include <array>

void Scene::Load(const char path[]) {

	std::array<uint8_t, 64 * 64 * 64> scene;

	for (int x = 0; x < 64; x++) {
		for (int y = 0; y < 64; y++) {
			for (int z = 0; z < 64; z++) {
				scene[x + y * 64 + z * 64 * 64] = sqrt(x * x + y * y + z * z) < 32;
			}
		}
	}

	cuda(Malloc(&voxels, 64 * 64 * 64));
	cuda(Memcpy(&voxels[0], &scene[0], 64 * 64 * 64, cudaMemcpyKind::cudaMemcpyHostToDevice));


	//const aiScene* importer_scene = importer.ReadFile(
	//	path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices);
	//StaticMesh mesh;
	//std::cout << "Loading scene:" << path << "\n";
	//mesh.load(importer_scene);
	//for (auto& vertex : mesh.vertices) {
	//	glm::vec3 tempVertex = glm::vec3{ vertex.x, vertex.z, vertex.y };
	//	vertex = tempVertex;
	//}
	////scale
	//for (auto& vertex : mesh.vertices) {
	//	vertex = vertex * glm::vec3{ 1, 1, 1 };
	//}
	//meshes.push_back(mesh);

	///*Compute the primitives and bounding boxes needed for BVH from the meshes*/;
	//for (const auto& mesh : meshes) {
	//	for (const auto& face : mesh.faces) {
	//		/*Load actual vertexes from mesh*/

	//		glm::vec3 vertexes[3];
	//		vertexes[0] = mesh.vertices[face.x];
	//		vertexes[1] = mesh.vertices[face.y];
	//		vertexes[2] = mesh.vertices[face.z];

	//		/*Compute bbox*/

	//		BBox bbox;
	//		for (const auto& vertex : vertexes) {
	//			bbox.addVertex(vertex);
	//		}
	//		primitiveBBoxes.push_back(bbox);

	//		/*Get primitive in the format we like*/

	//		Triangle tempTriangle;
	//		tempTriangle.vert = vertexes[0];
	//		tempTriangle.e1 = vertexes[1] - vertexes[0];
	//		tempTriangle.e2 = vertexes[2] - vertexes[0];
	//		// tempTriangle.materialType = mesh.reflectiveType;
	//		// tempTriangle.color = mesh.color;
	//		primitives.push_back(tempTriangle);
	//	}
	//}

	//if (primitives.size() == 0) {
	//	std::cerr << "No primitives found in scene, loading scene without any \n";
	//	return;
	//}
	//BVH bvh(primitives, primitiveBBoxes, PartitionAlgorithm::SAH);

	//CachedBVH cachedBVH;

	//cuda(Malloc(&cachedBVH.primitives, primitives.size() * sizeof(Triangle)));
	//cuda(Memcpy(&cachedBVH.primitives[0], &primitives[0],
	//			primitives.size() * sizeof(Triangle),
	//			cudaMemcpyKind::cudaMemcpyHostToDevice));

	//cuda(Malloc(&cachedBVH.nodes, bvh.nNodes * sizeof(BVH::BVHNode)));
	//cuda(Memcpy(cachedBVH.nodes, bvh.nodes.data(),
	//			bvh.nNodes * sizeof(BVH::BVHNode),
	//			cudaMemcpyKind::cudaMemcpyHostToDevice));

	//gpuScene.CUDACachedBVH = cachedBVH;
}
