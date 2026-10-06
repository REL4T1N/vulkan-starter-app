#include "application.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

namespace application {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// Матрицы хранятся в column-major порядке: m[column * 4 + row],
struct Vec3 {
	float x, y, z;
};

struct Mat4 {
	// Column-major storage: m[column * 4 + row]
	float m[16];
};

Mat4 matIdentity() {
	return { .m = {
		1.f, 0.f, 0.f, 0.f,
		0.f, 1.f, 0.f, 0.f,
		0.f, 0.f, 1.f, 0.f,
		0.f, 0.f, 0.f, 1.f,
	}};
}

// Умножение матриц: r = a * b (применение: сначала b, потом a)
Mat4 matMultiply(const Mat4& a, const Mat4& b) {
	Mat4 r{};
	for (int c = 0; c < 4; ++c) {
		for (int row = 0; row < 4; ++row) {
			float s = 0.f;
			for (int k = 0; k < 4; ++k) {
				s += a.m[k * 4 + row] * b.m[c * 4 + k];
			}
			r.m[c * 4 + row] = s;
		}
	}
	return r;
}

Mat4 matTranslate(float x, float y, float z) {
	Mat4 r = matIdentity();
	r.m[12] = x;
	r.m[13] = y;
	r.m[14] = z;
	return r;
}

Mat4 matScale(float x, float y, float z) {
	Mat4 r = matIdentity();
	r.m[0] = x;
	r.m[5] = y;
	r.m[10] = z;
	return r;
}

Mat4 matRotateX(float degrees) {
	const float a = degrees * kPi / 180.f;
	const float c = std::cos(a), s = std::sin(a);
	return { .m = {
		1.f, 0.f, 0.f, 0.f,
		0.f, c, s, 0.f,
		0.f, -s, c, 0.f,
		0.f, 0.f, 0.f, 1.f,
	}};
}

Mat4 matRotateY(float degrees) {
	const float a = degrees * kPi / 180.f;
	const float c = std::cos(a), s = std::sin(a);
	return { .m = {
		c, 0.f, -s, 0.f,
		0.f, 1.f, 0.f, 0.f,
		s, 0.f, c, 0.f,
		0.f, 0.f, 0.f, 1.f,
	}};
}

Mat4 matRotateZ(float degrees) {
	const float a = degrees * kPi / 180.f;
	const float c = std::cos(a), s = std::sin(a);
	return { .m = {
		c, s, 0.f, 0.f,
		-s, c, 0.f, 0.f,
		0.f, 0.f, 1.f, 0.f,
		0.f, 0.f, 0.f, 1.f,
	}};
}

// Перспективная проекция
Mat4 matPerspective(float fovYDegrees, float aspect, float zNear, float zFar) {
	const float f = 1.f / std::tan(fovYDegrees * kPi / 360.f);
	Mat4 r{};
	r.m[0] = f / aspect;
	r.m[5] = f;
	r.m[10] = (zFar + zNear) / (zNear - zFar);
	r.m[11] = -1.f;
	r.m[14] = 2.f * zFar * zNear / (zNear - zFar);
	return r;
}

// Ортографическая проекция: параллельные лучи, без схлопывания к горизонту
Mat4 matOrtho(float left, float right, float bottom, float top, float zNear, float zFar) {
	Mat4 r = matIdentity();
	r.m[0] = 2.f / (right - left);
	r.m[5] = 2.f / (top - bottom);
	r.m[10] = -2.f / (zFar - zNear);
	r.m[12] = -(right + left) / (right - left);
	r.m[13] = -(top + bottom) / (top - bottom);
	r.m[14] = -(zFar + zNear) / (zFar - zNear);
	return r;
}

// Матрица камеры: взгляд из точки eye в точку center
Mat4 matLookAt(Vec3 eye, Vec3 center, Vec3 up) {
	const auto sub = [](Vec3 a, Vec3 b) { return Vec3{ a.x - b.x, a.y - b.y, a.z - b.z }; };
	const auto cross = [](Vec3 a, Vec3 b) {
		return Vec3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
	};
	const auto dot = [](Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; };
	const auto norm = [dot](Vec3 a) {
		const float l = std::sqrt(dot(a, a));
		return Vec3{ a.x / l, a.y / l, a.z / l };
	};

	const Vec3 f = norm(sub(center, eye)); // направление взгляда
	const Vec3 s = norm(cross(f, up));     // "право" камеры
	const Vec3 u = cross(s, f);            // "верх" камеры

	return { .m = {
		s.x, u.x, -f.x, 0.f,
		s.y, u.y, -f.y, 0.f,
		s.z, u.z, -f.z, 0.f,
		-dot(s, eye), -dot(u, eye), dot(f, eye), 1.f,
	}};
}

// Конверсия OpenGL-проекции в clip space Vulkan:
//   1) инверсия Y (ось Y направлена вниз);
//   2) сжатие глубины из [-1, 1] в [0, 1] (z' = 0.5*z + 0.5*w).
Mat4 matVulkanClipCorrection() {
	return { .m = {
		1.f, 0.f, 0.f, 0.f,
		0.f, -1.f, 0.f, 0.f,
		0.f, 0.f, 0.5f, 0.f,
		0.f, 0.f, 0.5f, 1.f,
	}};
}

struct Vertex {
	Vec3 position;
	Vec3 color;
};

const Vec3 kTruncatedTetrahedronPositions[12] = {
	{ 3.f, 1.f, 1.f },   // 0  P(A,B)
	{ 1.f, 3.f, 1.f },   // 1  P(A,C)
	{ 1.f, 1.f, 3.f },   // 2  P(A,D)
	{ 3.f, -1.f, -1.f }, // 3  P(B,A)
	{ 1.f, -1.f, -3.f }, // 4  P(B,C)
	{ 1.f, -3.f, -1.f }, // 5  P(B,D)
	{ -1.f, 3.f, -1.f }, // 6  P(C,A)
	{ -1.f, 1.f, -3.f }, // 7  P(C,B)
	{ -3.f, 1.f, -1.f }, // 8  P(C,D)
	{ -1.f, -1.f, 3.f }, // 9  P(D,A)
	{ -1.f, -3.f, 1.f }, // 10 P(D,B)
	{ -3.f, -1.f, 1.f }, // 11 P(D,C)
};

// Индексы: треугольные грани + шестиугольники как веера треугольников.
// Обход против часовой стрелки при взгляде снаружи.
const uint16_t kTruncatedTetrahedronIndices[60] = {
	// треугольные грани (срезы вершин тетраэдра)
	0, 1, 2,
	3, 5, 4,
	6, 7, 8,
	9, 10, 11,
	// шестиугольник ABC
	0, 3, 4,
	0, 4, 7,
	0, 7, 6,
	0, 6, 1,
	// шестиугольник ABD
	0, 2, 9,
	0, 9, 10,
	0, 10, 5,
	0, 5, 3,
	// шестиугольник ACD
	1, 6, 8,
	1, 8, 11,
	1, 11, 9,
	1, 9, 2,
	// шестиугольник BCD
	4, 5, 10,
	4, 10, 11,
	4, 11, 8,
	4, 8, 7,
};

constexpr uint32_t kTruncatedTetrahedronIndexCount =
	sizeof(kTruncatedTetrahedronIndices) / sizeof(kTruncatedTetrahedronIndices[0]);

struct UniformBlock {
	float mvp[16];
	float color[4];
};

// Ресурсы одного объекта: геометрия + uniform-буфер + дескриптор
struct ObjectResources {
	VkBuffer vertex_buffer = VK_NULL_HANDLE;
	VmaAllocation vertex_allocation = VK_NULL_HANDLE;

	VkBuffer index_buffer = VK_NULL_HANDLE;
	VmaAllocation index_allocation = VK_NULL_HANDLE;

	VkBuffer uniform_buffer = VK_NULL_HANDLE;
	VmaAllocation uniform_allocation = VK_NULL_HANDLE;
	void* uniform_mapped = nullptr;

	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

	uint32_t index_count = 0;
};

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
ObjectResources objects[2];


int projection_index = 0; 

float figure_position[3] = { 0.f, 0.f, 0.f };
float figure_rotation[3] = { 0.f, 0.f, 0.f }; // градусы
float figure_scale[3] = { 1.f, 1.f, 1.f };
float figure_color[3] = { 1.f, 1.f, 1.f };

bool animation_enabled = true;
float animation_speed = 1.f;
float trajectory_radius = 2.f;
double animation_time = 0.0;
double last_frame_time = 0.0;

float second_position[3] = { 0.f, 0.f, 0.f };
float second_rotation[3] = { 0.f, 0.f, 0.f };
float second_scale[3] = { 0.5f, 0.5f, 0.5f };
float second_color[3] = { 1.f, 0.55f, 0.15f };

bool loadShader(const char* path, std::vector<char>& out) {
	std::ifstream file(path, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open shader file: " << path << '\n';
		return false;
	}

	const size_t size = size_t(file.tellg());
	out.resize(size);
	file.seekg(0);
	file.read(out.data(), std::streamsize(size));

	return true;
}

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer* buffer,
				  VmaAllocation* allocation, void** mapped) {
	auto& ctx = graphics::internal::context;

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	VmaAllocationInfo allocation_info{};

	const VmaAllocationCreateInfo allocation_create = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
				 VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
	};

	if (vmaCreateBuffer(ctx.allocator, &buffer_info, &allocation_create,
						buffer, allocation, &allocation_info) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan buffer\n";
		return false;
	}

	*mapped = allocation_info.pMappedData;
	return true;
}

bool createObjectResources(uint32_t object_index, const Vertex* vertices,
						   uint32_t vertex_count, const uint16_t* indices,
						   uint32_t index_count) {
	auto& ctx = graphics::internal::context;
	ObjectResources& object = objects[object_index];

	object.index_count = index_count;

	void* mapped = nullptr;

	if (!createBuffer(sizeof(Vertex) * vertex_count,
					  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
					  &object.vertex_buffer, &object.vertex_allocation, &mapped)) {
		return false;
	}
	std::memcpy(mapped, vertices, sizeof(Vertex) * vertex_count);

	if (!createBuffer(sizeof(uint16_t) * index_count,
					  VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
					  &object.index_buffer, &object.index_allocation, &mapped)) {
		return false;
	}
	std::memcpy(mapped, indices, sizeof(uint16_t) * index_count);

	if (!createBuffer(sizeof(UniformBlock), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
					  &object.uniform_buffer, &object.uniform_allocation,
					  &object.uniform_mapped)) {
		return false;
	}

	const VkDescriptorBufferInfo buffer_info = {
		.buffer = object.uniform_buffer,
		.offset = 0,
		.range = sizeof(UniformBlock),
	};

	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = object.descriptor_set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);

	return true;
}

bool initializeGraphicsPipeline() {
	auto& ctx = graphics::internal::context;

	const VkDescriptorSetLayoutBinding layout_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &layout_binding,
	};

	if (vkCreateDescriptorSetLayout(ctx.device, &layout_info, nullptr,
									&descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set layout\n";
		return false;
	}

	const VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};

	if (vkCreatePipelineLayout(ctx.device, &pipeline_layout_info, nullptr,
							   &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		return false;
	}

	std::vector<char> vert_code;
	std::vector<char> frag_code;

	if (!loadShader("shaders/triangle.vert.spv", vert_code) ||
	    !loadShader("shaders/triangle.frag.spv", frag_code)) {
		return false;
	}

	VkShaderModuleCreateInfo module_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
	};

	module_info.codeSize = vert_code.size();
	module_info.pCode = reinterpret_cast<const uint32_t*>(vert_code.data());

	VkShaderModule vertex_module;
	if (vkCreateShaderModule(ctx.device, &module_info, nullptr, &vertex_module) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan vertex shader module\n";
		return false;
	}

	module_info.codeSize = frag_code.size();
	module_info.pCode = reinterpret_cast<const uint32_t*>(frag_code.data());

	VkShaderModule fragment_module;
	if (vkCreateShaderModule(ctx.device, &module_info, nullptr, &fragment_module) != VK_SUCCESS) {
		vkDestroyShaderModule(ctx.device, vertex_module, nullptr);
		std::cerr << "Failed to create Vulkan fragment shader module\n";
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_module,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_module,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, color),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.f,
	};

	const VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
		.minDepthBounds = 0.f,
		.maxDepthBounds = 1.f,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
						  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = sizeof(stages) / sizeof(stages[0]),
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = ctx.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE,
													  1, &pipeline_info, nullptr, &pipeline);

	vkDestroyShaderModule(ctx.device, vertex_module, nullptr);
	vkDestroyShaderModule(ctx.device, fragment_module, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan graphics pipeline\n";
		return false;
	}

	return true;
}

bool initializeObjects() {
	auto& ctx = graphics::internal::context;

	std::array<Vertex, 12> vertices{};
	for (uint32_t i = 0; i < 12; ++i) {
		const Vec3& p = kTruncatedTetrahedronPositions[i];
		vertices[i].position = p;
		vertices[i].color = {
			0.5f + p.x / 6.f,
			0.5f + p.y / 6.f,
			0.5f + p.z / 6.f,
		};
	}

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 2,
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 2,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr,
							   &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor pool\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};

	for (uint32_t i = 0; i < 2; ++i) {
		if (vkAllocateDescriptorSets(ctx.device, &set_info,
									 &objects[i].descriptor_set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate Vulkan descriptor set #" << i << '\n';
			return false;
		}
	}

	if (!createObjectResources(0, vertices.data(), 12,
							   kTruncatedTetrahedronIndices, kTruncatedTetrahedronIndexCount)) {
		return false;
	}

	if (!createObjectResources(1, vertices.data(), 12,
							   kTruncatedTetrahedronIndices, kTruncatedTetrahedronIndexCount)) {
		return false;
	}

	return true;
}

Vec3 trajectory(double time, float radius, float phase) {
	return Vec3{
		radius * static_cast<float>(std::sin(time * 1.0 + phase)),
		radius * 0.5f * static_cast<float>(std::sin(time * 2.0 + phase)),
		radius * static_cast<float>(std::cos(time * 1.0 + phase)),
	};
}

void updateObjectUniform(uint32_t object_index, const Mat4& mvp, const float color[3]) {
	UniformBlock block{};
	std::memcpy(block.mvp, mvp.m, sizeof(block.mvp));
	block.color[0] = color[0];
	block.color[1] = color[1];
	block.color[2] = color[2];
	block.color[3] = 1.f;

	ObjectResources& object = objects[object_index];
	std::memcpy(object.uniform_mapped, &block, sizeof(block));
}

}

bool initialize() {
	if (!initializeGraphicsPipeline()) {
		return false;
	}

	if (!initializeObjects()) {
		return false;
	}

	return true;
}

void shutdown() {
	auto& ctx = graphics::internal::context;
	vkQueueWaitIdle(ctx.graphics_queue);

	vkDestroyPipeline(ctx.device, pipeline, nullptr);
	vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
	vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(ctx.device, descriptor_set_layout, nullptr);

	for (ObjectResources& object : objects) {
		vmaDestroyBuffer(ctx.allocator, object.vertex_buffer, object.vertex_allocation);
		vmaDestroyBuffer(ctx.allocator, object.index_buffer, object.index_allocation);
		vmaDestroyBuffer(ctx.allocator, object.uniform_buffer, object.uniform_allocation);
	}
}

void update(double time) {
	const double delta = time - last_frame_time;
	last_frame_time = time;

	if (animation_enabled) {
		animation_time += delta * animation_speed;
	}

	ImGui::Begin("Lab 1: Truncated Tetrahedron");

	// Задание 1: переключение проекции
	const char* projections[] = { "Perspective", "Orthographic" };
	ImGui::Combo("Projection", &projection_index, projections, IM_ARRAYSIZE(projections));

	// Задание 2: позиция, поворот, масштаб
	ImGui::DragFloat3("Position", figure_position, 0.05f, -10.f, 10.f, "%.2f");
	ImGui::DragFloat3("Rotation (deg)", figure_rotation, 1.f, -180.f, 180.f, "%.1f");
	ImGui::DragFloat3("Scale", figure_scale, 0.02f, 0.05f, 5.f, "%.2f");

	// Задание 3: пауза/воспроизведение, скорость, радиус траектории
	ImGui::Separator();
	ImGui::Checkbox("Animate", &animation_enabled);
	ImGui::SliderFloat("Speed", &animation_speed, 0.f, 3.f, "%.2f");
	ImGui::SliderFloat("Trajectory radius", &trajectory_radius, 0.f, 5.f, "%.2f");

	// Задание 4: цвет фигуры (умножается на цвета вершин — задание 5)
	ImGui::Separator();
	ImGui::ColorEdit3("Figure color", figure_color);

	// Задание 6: вторая фигура со своим набором дескрипторов
	if (ImGui::CollapsingHeader("Second object")) {
		ImGui::DragFloat3("Position##2", second_position, 0.05f, -10.f, 10.f, "%.2f");
		ImGui::DragFloat3("Rotation##2", second_rotation, 1.f, -180.f, 180.f, "%.1f");
		ImGui::DragFloat3("Scale##2", second_scale, 0.02f, 0.05f, 5.f, "%.2f");
		ImGui::ColorEdit3("Color##2", second_color);
	}

	ImGui::Text("Vertices: 12, Faces: 8 (4 hex + 4 tri)");
	ImGui::Text("Objects: 2 (separate descriptor sets)");

	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	auto& ctx = graphics::internal::context;

	if (fd.command_buffer == VK_NULL_HANDLE || fd.framebuffer == VK_NULL_HANDLE) {
		return;
	}

	const float aspect = float(ctx.swapchain_extent.width) /
						 float(ctx.swapchain_extent.height);

	// Доп. задание 1: перспективная или ортографическая проекция
	const Mat4 projection = projection_index == 0
		? matPerspective(50.f, aspect, 0.1f, 100.f)
		: matOrtho(-6.f * aspect, 6.f * aspect, -6.f, 6.f, 0.1f, 100.f);

	const Mat4 view = matLookAt(Vec3{ 0.f, 3.f, 9.f }, Vec3{ 0.f, 0.f, 0.f },
								Vec3{ 0.f, 1.f, 0.f });

	const Mat4 vulkan_clip = matVulkanClipCorrection();
	const Mat4 view_projection = matMultiply(vulkan_clip, matMultiply(projection, view));

	// Объект 0: модель = T * Ry * Rx * Rz * S.
	// К трансформациям пользователя добавляется смещение по траектории
	// и собственное вращение фигуры (задание 3)
	const Vec3 trajectory_offset = trajectory(animation_time, trajectory_radius, 0.0);
	const Vec3 position = {
		figure_position[0] + trajectory_offset.x,
		figure_position[1] + trajectory_offset.y,
		figure_position[2] + trajectory_offset.z,
	};

	Mat4 model = matMultiply(matTranslate(position.x, position.y, position.z),
							 matMultiply(matRotateY(figure_rotation[1] + float(animation_time) * 90.f),
							 matMultiply(matRotateX(figure_rotation[0]),
							 matMultiply(matRotateZ(figure_rotation[2]),
										 matScale(figure_scale[0], figure_scale[1], figure_scale[2])))));

	const Mat4 mvp0 = matMultiply(view_projection, model);
	updateObjectUniform(0, mvp0, figure_color);

	// Объект 1: вторая фигура, независимая траектория и обратное вращение
	const Vec3 trajectory1 = trajectory(animation_time, trajectory_radius * 0.7f, 2.4);
	const Vec3 position1 = {
		second_position[0] + trajectory1.x,
		second_position[1] + trajectory1.y,
		second_position[2] + trajectory1.z,
	};

	Mat4 model1 = matMultiply(matTranslate(position1.x, position1.y, position1.z),
							  matMultiply(matRotateY(second_rotation[1] - float(animation_time) * 120.f),
							  matMultiply(matRotateX(second_rotation[0] + float(animation_time) * 60.f),
							  matMultiply(matRotateZ(second_rotation[2]),
										  matScale(second_scale[0], second_scale[1], second_scale[2])))));

	const Mat4 mvp1 = matMultiply(view_projection, model1);
	updateObjectUniform(1, mvp1, second_color);

	// Запись команд отрисовки
	vkResetCommandBuffer(fd.command_buffer, 0);

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	if (vkBeginCommandBuffer(fd.command_buffer, &begin) != VK_SUCCESS) {
		std::cerr << "Failed to begin Vulkan command buffer\n";
		return;
	}

	const VkClearValue clear_values[] = {
		{ .color = { { 0.08f, 0.09f, 0.11f, 1.f } } },
		{ .depthStencil = { 1.f, 0 } },
	};

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = ctx.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .offset = { 0, 0 }, .extent = ctx.swapchain_extent },
		.clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	const VkViewport viewport = {
		0.f, 0.f,
		float(ctx.swapchain_extent.width), float(ctx.swapchain_extent.height),
		0.f, 1.f,
	};

	const VkRect2D scissor = {
		.offset = { 0, 0 },
		.extent = ctx.swapchain_extent,
	};

	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	// Отрисовка обоих объектов: у каждого свой VkDescriptorSet (адание 6)
	for (const ObjectResources& object : objects) {
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
								pipeline_layout, 0, 1, &object.descriptor_set, 0, nullptr);

		const VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &object.vertex_buffer, &offset);
		vkCmdBindIndexBuffer(fd.command_buffer, object.index_buffer, 0,
							 VK_INDEX_TYPE_UINT16);

		vkCmdDrawIndexed(fd.command_buffer, object.index_count, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(fd.command_buffer);

	if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to record Vulkan command buffer\n";
	}
}

} // namespace application
