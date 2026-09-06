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

bool PerformanceMeasure::measure_convergence(const double delta) {
	constexpr double warmup_seconds = 1.0;
	constexpr double measure_seconds = 3.0;

	static int current_test = 0;
	static double phase_time = 0.0;
	static std::vector<float> times;

	camera.position = test_positions[current_test];
	camera.horizontal_angle = test_angles[current_test].x;
	camera.vertical_angle = test_angles[current_test].y;

	phase_time += delta;
	if (phase_time <= warmup_seconds) {
		return false;
	}

	times.push_back((float)delta);

	if (phase_time < warmup_seconds + measure_seconds) {
		return false;
	}

	const double average_delta = std::accumulate(times.begin(), times.end(), 0.f) / times.size();
	const auto [mn, mx] = std::minmax_element(times.begin(), times.end());

	file << "View " << current_test << "\n";
	file << "Average ms: " << average_delta * 1000.0 << "\n";
	file << "Average fps: " << 1.0 / average_delta << "\n";
	file << "Min ms: " << *mn * 1000.0 << "\n";
	file << "Max ms: " << *mx * 1000.0 << "\n";
	file << "Frames: " << times.size() << "\n\n";
	file.flush();

	times.clear();
	phase_time = 0.0;

	if (current_test < test_positions.size() - 1) {
		current_test++;
		return false;
	}
	return true;
}