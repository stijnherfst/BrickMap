#include "stdafx.h"

#include "IMGUI/imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

//#define PERFORMANCE_TEST

static void glfw_error_callback(int error, const char* description) {
	std::cout << description << "\n";
}

static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}
}

#ifdef PERFORMANCE_TEST
PerformanceMeasure performance;
#endif

int main(int argc, char* argv[]) {
	// OpenGL and GLFW setup
	GLFWwindow* window;

	glfwSetErrorCallback(glfw_error_callback);

	if (!glfwInit())
		exit(EXIT_FAILURE);

	glfwWindowHint(GLFW_DEPTH_BITS, 0);
	glfwWindowHint(GLFW_STENCIL_BITS, 0);
	glfwWindowHint(GLFW_RED_BITS, 32);
	glfwWindowHint(GLFW_GREEN_BITS, 32);
	glfwWindowHint(GLFW_BLUE_BITS, 32);
	glfwWindowHint(GLFW_ALPHA_BITS, 32);

	glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(render_width, render_height, "CUDA Path Tracer", nullptr, nullptr);

	if (!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(window);


	// Center cursor
	int w, h;
	glfwGetWindowSize(window, &w, &h);
	glfwSetCursorPos(window, w * 0.5, h * 0.5);

	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

	// ignore vsync
	glfwSwapInterval(0);

	// only copy r/g/b
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);

	// IMGUI setup
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	(void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init();

	// Setup style
	ImGui::StyleColorsDark();

	// CUDA setup
	cudaError_t cuda_err;

	int gl_device_id;
	unsigned int gl_device_count;

	cuda_err = cuda(GLGetDevices(&gl_device_count, &gl_device_id, 1u, cudaGLDeviceListAll));

	int cuda_device_id = (argc > 1) ? atoi(argv[1]) : gl_device_id;
	cuda_err = cuda(SetDevice(cuda_device_id));

	const bool multi_gpu = gl_device_id != cuda_device_id;
	struct cudaDeviceProp props;

	cuda_err = cuda(GetDeviceProperties(&props, gl_device_id));
	printf("GL   : %-24s (%2d)\n", props.name, props.multiProcessorCount);

	cuda_err = cuda(GetDeviceProperties(&props, cuda_device_id));
	printf("CUDA : %-24s (%2d)\n", props.name, props.multiProcessorCount);
	sm_cores = props.multiProcessorCount;

	cuda_interop interop;

	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	cuda_err = interop.set_size(width, height);

	glfwSetKeyCallback(window, glfw_key_callback);

	Scene scene;
	scene.generate();

	// Allocate ray queue buffer
	RayQueue* ray_buffer_work;
	cuda(Malloc(&ray_buffer_work, ray_queue_buffer_size * sizeof(RayQueue)));

	RayQueue* ray_buffer_next;
	cuda(Malloc(&ray_buffer_next, ray_queue_buffer_size * sizeof(RayQueue)));

	ShadowQueue* shadow_queue_buffer;
	cuda(Malloc(&shadow_queue_buffer, ray_queue_buffer_size * sizeof(ShadowQueue)));

	glm::vec4* blit_buffer;
	cuda(Malloc(&blit_buffer, render_width * render_height * sizeof(glm::vec4)));

	double previous_time = glfwGetTime();
	while (!glfwWindowShouldClose(window)) {
		double delta = glfwGetTime() - previous_time;
		previous_time = glfwGetTime();

		if (glfwGetKey(window, GLFW_KEY_MINUS)) {
			sun_position += glm::vec2(0.05 * delta, 0.05 * delta);
			sun_position_changed = true;
		}

		if (glfwGetKey(window, GLFW_KEY_EQUAL)) {
			sun_position -= glm::vec2(0.05 * delta, 0.05 * delta);
			sun_position_changed = true;
		}

#ifdef PERFORMANCE_TEST
		bool done = performance.measure(delta);
		if (done) {
			break;
		}
#else
		camera.handle_input(window, delta);
#endif

		camera.update();

		launch_kernels(interop.ca, blit_buffer, scene.gpuScene, ray_buffer_work, ray_buffer_next, shadow_queue_buffer);

		scene.process_load_queue();

		std::swap(ray_buffer_work, ray_buffer_next);
		interop.blit();

		// IMGUI
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		{
			static std::vector<float> frame_times;

			frame_times.push_back(delta);

			if (frame_times.size() > 200) {
				frame_times.erase(frame_times.begin());
			}

			ImGui::Begin("Performance");
			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGui::PlotHistogram("", frame_times.data(), frame_times.size(), 0, "Frametimes", 0, FLT_MAX, { 400, 100 });
			ImGui::Text("X: %f, Y: %f, Z: %f", camera.position.x, camera.position.y, camera.position.z);
			ImGui::Text("Hor: %f, Vert: %f", camera.horizontal_angle, camera.vertical_angle);
			ImGui::Text("Sun X: %f Y: %f", sun_position.x, sun_position.y);
			ImGui::SliderFloat("FocalDistance", &camera.focalDistance, 0.1f, 100.f);
			ImGui::SliderFloat("LensRadius", &camera.lensRadius, 0.01f, 80, "%.3f", 1.3f);
			//TODO(Dan): Backspace doesn't trigger. Manually need to call ImGui backend?
			ImGui::InputFloat("Lens Rad", &camera.lensRadius);

			ImGui::End();
		}

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	scene.dump();
	glfwDestroyWindow(window);
	glfwTerminate();

	cuda(DeviceReset());

	exit(EXIT_SUCCESS);
}