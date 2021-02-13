#pragma once

struct PerformanceMeasure {
	std::vector<glm::vec3> test_positions = { 
		{ 512, 512, 300 },
		{ 840.254, 832.446, 1169.88 },
		{ 2227.83, 774.886, 204.955 },
		{ 3326.19, 2055.72, 44.7995 },
		{ 7134.6, 1262.44, 5531.79 },
		{ 11298.6, 3113.03, 598.019 },
		{ 10921.4, 4774.14, 267.808 },
		{ 9961.29, 4508.12, 189.59 },
		{ 10835.3, 4160.83, 359.992 }
	};

	std::vector<glm::vec2> test_angles = { 
		{ -61863.5, -0.501796 },
		{ -61864.4, -0.429796 },
		{ -61863.9, 0.0622036 },
		{ -61864.2, -0.981796 },
		{ -61865.2, -0.501796 },
		{ -61866.3, -0.141796 },
		{ -61859.4, 0.0142036 },
		{ -61857.2, -0.261796 }
	};

	std::ofstream file;


	PerformanceMeasure();

	/// Measures the performance by cycling through some scenes. Will return true if measuring is over
	bool measure_path(double delta);
	bool measure_convergence(double delta);
};