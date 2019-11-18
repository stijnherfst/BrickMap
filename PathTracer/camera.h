#pragma once

struct Camera {
	glm::vec3 position = { 48, 48, 48 };
	glm::vec3 direction = { 1, 0, 0 };
	glm::vec3 up = { 0, 0, 1 };

	float focalDistance = 1;
	float lensRadius = 0.0f;
	bool enableDoF = false;

	double fov = 70;
	double aspect_ratio = render_width / render_height;
	double fov_rad = (pi / 180.0) * static_cast<double>(fov); // Need radians
	double tan_height = 2.0 * tan(fov_rad * 0.5);

	double horizontal_angle = 0.0;
	double vertical_angle = 0.0;

	void handle_input(GLFWwindow* window, double delta);
	void update();
};

extern Camera camera;