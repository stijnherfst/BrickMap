#include "stdafx.h"

void Camera::handle_input(GLFWwindow* window, double delta) {
	float speed = 10;
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)) {
		speed = 400;
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

	if (glfwGetKey(window, GLFW_KEY_F5)) {
		std::cout << position.x << ", " << position.y << ", " << position.z << "\n"
				  << horizontal_angle << ", " << vertical_angle << "\n";
	}

	double x, y;
	glfwGetCursorPos(window, &x, &y);
	int w, h;
	glfwGetWindowSize(window, &w, &h);

	aspect_ratio = static_cast<float>(w) / static_cast<float>(h);

	horizontal_angle = x / w * 10.0;
	vertical_angle = -y / h * 10.0;
	vertical_angle = std::max(-pi / 2.0 + 0.001, std::min(vertical_angle, pi / 2.0 - 0.001));
}

void Camera::update() {
	direction = glm::vec3(std::cos(vertical_angle) * std::sin(horizontal_angle),
						  std::cos(vertical_angle) * std::cos(horizontal_angle),
						  std::sin(vertical_angle));

	direction = glm::normalize(direction);
}

Camera camera;