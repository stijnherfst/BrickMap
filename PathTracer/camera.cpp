#include "stdafx.h"

void Camera::handle_input(GLFWwindow* window, double delta) {
	float speed = 1;
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)) {
		speed = 40;
	}

	if (glfwGetKey(window, GLFW_KEY_W)) {
		position += direction * speed * float(delta);
	} else if (glfwGetKey(window, GLFW_KEY_S)) {
		position -= direction * speed * float(delta);
	}

	const glm::vec3 displacement = glm::normalize(glm::cross(direction, up)) * speed * float(delta);
	if (glfwGetKey(window, GLFW_KEY_A)) {
		position -= displacement;
	} else if (glfwGetKey(window, GLFW_KEY_D)) {
		position += displacement;
	}

	if (glfwGetKey(window, GLFW_KEY_SPACE)) {
		position.z += 1 * speed * float(delta);
	} else if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL)) {
		position.z -= 1 * speed * float(delta);
	}
	if (glfwGetKey(window, GLFW_KEY_LEFT_ALT)) {
		return;
	}

	double x, y;
	glfwGetCursorPos(window, &x, &y);
	int w, h;
	glfwGetWindowSize(window, &w, &h);

	const double diffx = x - w * 0.5;
	const double diffy = y - h * 0.5;

	horizontal_angle += diffx * 0.012;
	vertical_angle -= diffy * 0.012;
	vertical_angle = std::max(-pi / 2.0 + 0.001, std::min(vertical_angle, pi / 2.0 - 0.001));

	glfwSetCursorPos(window, w * 0.5, h * 0.5);
}

void Camera::update() {
	direction = glm::vec3(std::cos(vertical_angle) * std::sin(horizontal_angle),
						  std::cos(vertical_angle) * std::cos(horizontal_angle),
						  std::sin(vertical_angle));

	direction = glm::normalize(direction);
}

Camera camera;