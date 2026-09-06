#pragma once

struct PerformanceMeasure {
	std::vector<glm::vec3> test_positions = {
		{ 512, 512, 300 },
		{ 840.254, 832.446, 1169.88 },
		{ 2227.83, 774.886, 204.955 },
		{ 7134.6, 1262.44, 5531.79 }
	};

	std::vector<glm::vec2> test_angles = {
		{ -61863.5, -0.501796 },
		{ -61864.4, -0.429796 },
		{ -61863.9, 0.0622036 },
		{ -61865.2, -0.501796 }
	};

	std::ofstream file;

	PerformanceMeasure();

	/// Measures the performance by cycling through some scenes. Will return true if measuring is over
	bool measure_path(double delta);
	bool measure_convergence(double delta);
};