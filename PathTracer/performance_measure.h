#pragma once

struct PerformanceMeasure {
	std::vector<glm::vec3> test_positions = { 
		{ -0.119, -26.116, 32.537 }, 
		{ -52.741, -44.67, 109.04 }, 
		{ 74.65, 2.77, 17.336 } 
	};
	std::vector<glm::vec2> test_angles = { 
		{ 12.576, -0.518204 }, 
		{ -6470.568, -0.818204 }, 
		{ -10218.468, 0.081796 } 
	};

	std::ofstream file;


	PerformanceMeasure();

	/// Measures the performance by cycling through some scenes. Will return true if measuring is over
	bool measure(double delta);
};