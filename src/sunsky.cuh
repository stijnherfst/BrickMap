//// Atmospheric scattering model
////
//// IMPORTANT COPYRIGHT INFO:
//// -----------------------------------
//// The license of this fragment is not completely clear to me, but for all I
//// can tell this shader derives from the MIT licensed source given below.
////
//// This fragment derives from this shader: http://glsl.herokuapp.com/e#9816.0
//// written by Martijn Steinrucken: countfrolic@gmail.com
////
//// Which in turn contained the following copyright info:
//// Code adapted from Martins:
////
///http://blenderartists.org/forum/showthread.php?242940-unlimited-planar-reflections-amp-refraction-%28update%29
////
//// Which in turn originates from:
////
///https://github.com/SimonWallner/kocmoc-demo/blob/RTVIS/media/shaders/sky.frag
//// where it was MIT licensed:
//// https://github.com/SimonWallner/kocmoc-demo/blob/RTVIS/README.rst
//// Heavily altered by me
#pragma once

// Angular sun size - physical sun is 0.53 degrees
constexpr float sunSize = 1.5;

constexpr float cutoffAngle = pi / 1.95f;
constexpr float steepness = 1.5;
constexpr float SkyFactor = 1.f;
constexpr float turbidity = 1;
constexpr float mieCoefficient = 0.005f;
constexpr float mieDirectionalG = 0.80f;

constexpr float v = 4.0;

// optical length at zenith for molecules
constexpr float rayleighZenithLength = 8.4E3;
constexpr float mieZenithLength = 1.25E3;

constexpr float sunIntensity = 1000.0;

constexpr glm::vec3 primaryWavelengths = glm::vec3(680E-9, 550E-9, 450E-9);

__device__ extern glm::vec3 K;
__device__ extern glm::vec3 up;
__device__ extern float sunAngularDiameterCos;
__device__ extern glm::vec2 SunPos;
__device__ extern glm::vec3 sunDirection;

__device__ glm::vec3 ortho(glm::vec3 v);
__device__ glm::vec3 getConeSample(glm::vec3 dir, float extent,
                                   unsigned int& seed);

__device__ float RayleighPhase(float cosViewSunAngle);
__device__ glm::vec3 totalMie(glm::vec3 primaryWavelengths, glm::vec3 K,
                              float T);
__device__ float hgPhase(float cosViewSunAngle, float g);
__device__ float SunIntensity(float zenithAngleCos);
glm::vec3 fromSpherical(glm::vec2 p);

__device__ glm::vec3 sun(glm::vec3 viewDir);
__device__ glm::vec3 sky(glm::vec3 viewDir);
__device__ glm::vec3 sunsky(glm::vec3 viewDir);
