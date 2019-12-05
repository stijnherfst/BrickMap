#pragma once

struct PerformanceMeasure {
	std::vector<glm::vec3> test_positions = { 
		{ -247.048, 570.164, 1417.91 },
		{ 779.12, 778.23, 1873.53 },
		{ 2042.08, 1511.62, 1544.69 },
		{ 678.714, 1986.4, 1847.71 },
		{ 466.26, 1180.75, 3001.93 }
	};
	std::vector<glm::vec2> test_angles = { 
		{ 32.772, -0.225796 },
		{ 366632, -1.3658 },
		{ 366634, -0.321796 },
		{ 366633, -0.717796 },
		{ 366632, -1.2938 }
	};

	std::ofstream file;


	PerformanceMeasure();

	/// Measures the performance by cycling through some scenes. Will return true if measuring is over
	bool measure(double delta);
};