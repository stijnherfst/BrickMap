#include "stdafx.h"
#include "sunsky.cuh"

__device__ glm::vec3 K = glm::vec3(0.686, 0.678, 0.666);
__device__ glm::vec3 up = glm::vec3(0.0, 0.0, 1.0);
__device__ float sunAngularDiameterCos;
__device__ glm::vec2 SunPos;
__device__ glm::vec3 sunDirection;

__device__ float RayleighPhase(float cosViewSunAngle) {
	return (3.0 / (16.0 * pi)) * (1.0 + powf(cosViewSunAngle, 2.0));
}

__device__ glm::vec3 totalMie(glm::vec3 primaryWavelengths, glm::vec3 K,
							  float T) {
	float c = (0.2 * T) * 10E-18;
	return 0.434f * c * pi * glm::pow((2.0f * pi) / primaryWavelengths, glm::vec3(v - 2.0)) * K;
}

__device__ float hgPhase(float cosViewSunAngle, float g) {
	return (1.0 / (4.0 * pi)) * ((1.0 - powf(g, 2.0)) / pow(1.0 - 2.0 * g * cosViewSunAngle + powf(g, 2.0), 1.5));
}

__device__ float SunIntensity(float zenithAngleCos) {
	return sunIntensity * glm::max(0.0, 1.0 - exp(-((cutoffAngle - acos(zenithAngleCos)) / steepness)));
}

glm::vec3 fromSpherical(glm::vec2 p) {
	return glm::vec3(cos(p.x) * sin(p.y), sin(p.x) * sin(p.y), cos(p.y));
}

__device__ glm::vec3 sun(glm::vec3 viewDir) {
	// Cos Angles
	float cosViewSunAngle = dot(viewDir, sunDirection);
	float cosSunUpAngle = dot(sunDirection, up);
	float cosUpViewAngle = dot(up, viewDir);

	float sunE = SunIntensity(cosSunUpAngle); // Get sun intensity based on how high in
		// the sky it is extinction (absorption +
		// out scattering) rayleigh coeficients
	glm::vec3 rayleighAtX = glm::vec3(5.176821E-6, 1.2785348E-5, 2.8530756E-5);

	// mie coefficients
	glm::vec3 mieAtX = totalMie(primaryWavelengths, K, turbidity) * mieCoefficient;

	// optical length
	// cutoff angle at 90 to avoid singularity in next formula.
	float zenithAngle = glm::max(0.0f, cosUpViewAngle);

	float rayleighOpticalLength = rayleighZenithLength / zenithAngle;
	float mieOpticalLength = mieZenithLength / zenithAngle;

	// combined extinction factor
	glm::vec3 Fex = exp(-(rayleighAtX * rayleighOpticalLength + mieAtX * mieOpticalLength));

	// in scattering
	glm::vec3 rayleighXtoEye = rayleighAtX * RayleighPhase(cosViewSunAngle);
	glm::vec3 mieXtoEye = mieAtX * hgPhase(cosViewSunAngle, mieDirectionalG);

	glm::vec3 totalLightAtX = rayleighAtX + mieAtX;
	glm::vec3 lightFromXtoEye = rayleighXtoEye + mieXtoEye;

	glm::vec3 somethingElse = sunE * (lightFromXtoEye / totalLightAtX);

	glm::vec3 sky = somethingElse * (1.0f - Fex);
	sky *= mix(glm::vec3(1.0), pow(somethingElse * Fex, glm::vec3(0.5f)),
			   glm::clamp(pow(1.0f - dot(up, sunDirection), 5.0f), 0.0f, 1.0f));

	// composition + solar disc
	float sundisk = sunAngularDiameterCos < (cosViewSunAngle ? 1.0 : 0.0);
	glm::vec3 sun = (sunE * 19000.0f * Fex) * sundisk;

	return 0.01f * sun;
}

__device__ glm::vec3 sky(glm::vec3 viewDir) {
	// Cos Angles
	float cosViewSunAngle = dot(viewDir, sunDirection);
	float cosSunUpAngle = dot(sunDirection, up);
	float cosUpViewAngle = dot(up, viewDir);

	float sunE = SunIntensity(cosSunUpAngle); // Get sun intensity based on how high in
		// the sky it is extinction (asorbtion + out
		// scattering) rayleigh coeficients
	glm::vec3 rayleighAtX = glm::vec3(5.176821E-6, 1.2785348E-5, 2.8530756E-5);

	// mie coefficients
	glm::vec3 mieAtX = totalMie(primaryWavelengths, K, turbidity) * mieCoefficient;

	// optical length
	// cutoff angle at 90 to avoid singularity in next formula.
	float zenithAngle = glm::max(0.0f, cosUpViewAngle);

	float rayleighOpticalLength = rayleighZenithLength / zenithAngle;
	float mieOpticalLength = mieZenithLength / zenithAngle;

	// combined extinction factor
	glm::vec3 Fex = exp(-(rayleighAtX * rayleighOpticalLength + mieAtX * mieOpticalLength));

	// in scattering
	glm::vec3 rayleighXtoEye = rayleighAtX * RayleighPhase(cosViewSunAngle);
	glm::vec3 mieXtoEye = mieAtX * hgPhase(cosViewSunAngle, mieDirectionalG);

	glm::vec3 totalLightAtX = rayleighAtX + mieAtX;
	glm::vec3 lightFromXtoEye = rayleighXtoEye + mieXtoEye;

	glm::vec3 somethingElse = sunE * (lightFromXtoEye / totalLightAtX);

	glm::vec3 sky = somethingElse * (1.0f - Fex);
	sky *= mix(glm::vec3(1.0), glm::pow(somethingElse * Fex, glm::vec3(0.5f)),
			   glm::clamp(pow(1.0f - dot(up, sunDirection), 5.0f), 0.0f, 1.0f));

	return SkyFactor * 0.01f * sky;
}

__device__ glm::vec3 sunsky(glm::vec3 viewDir) {
	// Cos Angles
	float cosViewSunAngle = dot(viewDir, sunDirection);
	float cosSunUpAngle = dot(sunDirection, up);
	float cosUpViewAngle = dot(up, viewDir);
	if (sunAngularDiameterCos == 1.0f) {
		return glm::vec3(1.0f, 0.0f, 0.0f);
	}
	float sunE = SunIntensity(cosSunUpAngle); // Get sun intensity based on how high in
		// the sky it is extinction (asorbtion + out
		// scattering) rayleigh coeficients
	glm::vec3 rayleighAtX = glm::vec3(5.176821E-6f, 1.2785348E-5f, 2.8530756E-5f);

	// mie coefficients
	glm::vec3 mieAtX = totalMie(primaryWavelengths, K, turbidity) * mieCoefficient;

	// optical length
	// cutoff angle at 90 to avoid singularity in next formula.
	float zenithAngle = glm::max(0.0f, cosUpViewAngle);

	float rayleighOpticalLength = rayleighZenithLength / zenithAngle;
	float mieOpticalLength = mieZenithLength / zenithAngle;

	// combined extinction factor
	glm::vec3 Fex = exp(-(rayleighAtX * rayleighOpticalLength + mieAtX * mieOpticalLength));

	// in scattering
	glm::vec3 rayleighXtoEye = rayleighAtX * RayleighPhase(cosViewSunAngle);
	glm::vec3 mieXtoEye = mieAtX * hgPhase(cosViewSunAngle, mieDirectionalG);

	glm::vec3 totalLightAtX = rayleighAtX + mieAtX;
	glm::vec3 lightFromXtoEye = rayleighXtoEye + mieXtoEye;

	glm::vec3 somethingElse = sunE * (lightFromXtoEye / totalLightAtX);

	glm::vec3 sky = somethingElse * (1.0f - Fex);
	sky *= glm::mix(glm::vec3(1.0f), glm::pow(somethingElse * Fex, glm::vec3(0.5f)),
					glm::clamp(pow(1.0f - dot(up, sunDirection), 5.0f), 0.0f, 1.0f));

	// composition + solar disc
	float sundisk = glm::smoothstep(
		sunAngularDiameterCos, sunAngularDiameterCos + 0.00002f, cosViewSunAngle);
	glm::vec3 sun = (sunE * 19000.0f * Fex) * sundisk * 1E-5f;

	return 0.01f * (sun + sky);
}

__device__ glm::vec3 ortho(glm::vec3 v) {
	return abs(v.x) > abs(v.z) ? glm::vec3(-v.y, v.x, 0.0f)
							   : glm::vec3(0.0f, -v.z, v.y);
}

__device__ float RandomFloat2(unsigned int& seed);

__device__ glm::vec3 getConeSample(glm::vec3 dir, float extent,
								   unsigned int& seed) {
	// Create orthogonal vector (fails for z,y = 0)
	dir = normalize(dir);
	glm::vec3 o1 = glm::normalize(ortho(dir));
	glm::vec3 o2 = glm::normalize(glm::cross(dir, o1));

	// Convert to spherical coords aligned to dir
	glm::vec2 r = { RandomFloat2(seed), RandomFloat2(seed) };

	r.x = r.x * 2.f * pi;
	r.y = 1.0f - r.y * extent;

	float oneminus = sqrt(1.0f - r.y * r.y);
	return cos(r.x) * oneminus * o1 + sin(r.x) * oneminus * o2 + r.y * dir;
}