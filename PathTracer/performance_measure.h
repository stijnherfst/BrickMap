#pragma once

struct PerformanceMeasure {
	std::vector<glm::vec3> test_positions = { 
		{ 295, 33, 227 },
		{ -2, 103, 72 },
		{ 87, 193, -32 },
		{ 51, 236, 223 },
		{ 124, 186, 259 },
		{ 52, 277, 128 }
	};
	std::vector<glm::vec2> test_angles = { 
		{ -1.092, -0.48 },
		{ -1438.248, 0.441796 },
		{ 9377, 1.226204 },
		{ 4388.256, -0.525796 },
		{ 5211.744, -0.477796 },
		{ 13593.132, 0.057796 }
	};

	std::ofstream file;


	PerformanceMeasure();

	/// Measures the performance by cycling through some scenes. Will return true if measuring is over
	bool measure(double delta);
};