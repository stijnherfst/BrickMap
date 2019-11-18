#pragma once

class Scene {
public:
	struct GPUScene {
		CachedBVH CUDACachedBVH;
	} gpuScene;

	uint8_t* voxels;

	void Load(const char path[]);

private:
	// const aiScene *scene;
	/*Assimp::Importer importer;

	std::vector<Triangle> primitives;
	std::vector<BBox> primitiveBBoxes;*/
	// Sphere *spheres;
	//std::vector<StaticMesh> meshes;
};