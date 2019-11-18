#pragma once
//Header file intended to keep all the different types of rays we want to have.
//(i.e Differential rays? Secondary rays for certain objects?)

struct Ray {
	glm::vec3 orig;
	glm::vec3 dir;
	__device__ Ray(glm::vec3 origin, glm::vec3 direction)
		: orig(origin)
		, dir(direction) {}
};
