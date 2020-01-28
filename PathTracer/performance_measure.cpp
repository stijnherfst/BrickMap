#include "stdafx.h"

#include <filesystem>
namespace fs = std::filesystem;

PerformanceMeasure::PerformanceMeasure() {
	file.open("performance.txt", std::ios::app);
	file << "\n\n";
}

bool PerformanceMeasure::measure_path(double delta) {
	static double time_elapsed = 0.0;

	time_elapsed += delta;

	float viewpoint_interpolated = time_elapsed / 10000.0;

	//camera.position = test_positions[current_test];






	static int current_test = 0;
	static double last_time = glfwGetTime();
	static double delta_min = std::numeric_limits<double>::max();
	static double delta_max = 0;

	static std::vector<float> times;
	times.push_back((float)delta);

	camera.position = test_positions[current_test];
	camera.horizontal_angle = test_angles[current_test].x;
	camera.vertical_angle = test_angles[current_test].y;

	delta_min = std::min(delta_min, delta);
	delta_max = std::max(delta_max, delta);

	// Every 10 seconds
	if (glfwGetTime() - last_time > 10.f) {
		last_time = glfwGetTime();

		double average_delta = std::accumulate(times.begin(), times.end(), 0.f) / times.size();
		file << "Average ms: " << average_delta * 1000.0 << "\n";
		file << "Average fps: " << 1.0 / average_delta << "\n";
		file << "Min ms: " << delta_min * 1000.0 << "\n";
		file << "Max ms: " << delta_max * 1000.0 << "\n";

		delta_min = std::numeric_limits<double>::max();
		delta_max = 0;

		times.clear();
		if (current_test < test_positions.size() - 1) {
			current_test++;
			file << "\n";
		} else {
			return true;
		}
	}
	
	return false;
}

bool PerformanceMeasure::measure_convergence(double delta) {
	static int current_test = 0;
	static double last_time = glfwGetTime();
	static double delta_min = std::numeric_limits<double>::max();
	static double delta_max = 0;

	static std::vector<float> times;
	times.push_back((float)delta);

	camera.position = test_positions[current_test];
	camera.horizontal_angle = test_angles[current_test].x;
	camera.vertical_angle = test_angles[current_test].y;

	delta_min = std::min(delta_min, delta);
	delta_max = std::max(delta_max, delta);

	// Every 5 seconds
	if (glfwGetTime() - last_time > 5.f) {
		last_time = glfwGetTime();

		double average_delta = std::accumulate(times.begin(), times.end(), 0.f) / times.size();
		file << "Average ms: " << average_delta * 1000.0 << "\n";
		file << "Average fps: " << 1.0 / average_delta << "\n";
		file << "Min ms: " << delta_min * 1000.0 << "\n";
		file << "Max ms: " << delta_max * 1000.0 << "\n";

		delta_min = std::numeric_limits<double>::max();
		delta_max = 0;

		times.clear();
		if (current_test < test_positions.size() - 1) {
			current_test++;
			file << "\n";
		} else {
			return true;
		}
	}

	return false;
}