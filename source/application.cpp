#include "application.hpp"
#include "scene.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <imgui.h>

namespace application {
namespace {
using namespace scene;

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};
struct ObjectResources {
    Buffer vertices, indices, uniform;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    uint32_t index_count = 0;
};
struct ModelControls {
    float position[3];
    float rotation[3];
    float scale[3];
    float albedo[3];
    float specular[3] = {0.8f, 0.8f, 0.8f};
    float shininess = 32.f;
    bool vertex_colors = false;
};
struct SpotControls {
    PointLight light = {{0.f, 5.f, 5.f, 0.f}, {1.f, 0.85f, 0.6f, 35.f}};
    float pitch = -45.f, yaw = 0.f;
    float inner = 18.f, outer = 30.f;
};

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
VkDescriptorSetLayout scene_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_set = VK_NULL_HANDLE;
Buffer scene_buffer, point_buffer, spot_buffer;
ObjectResources objects[2];
VkDeviceSize max_storage_range = 0;
const char* resource_error = nullptr;

ModelControls models[2] = {
    {{-2.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.65f, 0.65f, 0.65f}, {0.25f, 0.55f, 0.85f}},
    {{ 2.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.5f, 0.5f, 0.5f}, {0.85f, 0.4f, 0.15f}},
};
float camera_position[3] = {0.f, 4.f, 13.f};
float camera_angles[3] = {-17.f, 0.f, 0.f}; // pitch, yaw, roll in degrees
float camera_fov = 50.f;
float ortho_size = 6.f;
int projection_index = 0;
float ambient_color[3] = {1.f, 1.f, 1.f};
float ambient_intensity = 0.08f;
float directional_angles[2] = {-45.f, 30.f};
float directional_color[3] = {1.f, 1.f, 1.f};
float directional_intensity = 0.7f;
std::vector<PointLight> point_lights = {{{-3.f, 3.f, 4.f, 0.f}, {0.6f, 0.8f, 1.f, 20.f}}};
std::vector<SpotControls> spot_lights = {SpotControls{}};
std::vector<SpotLight> packed_spots;

bool animation_enabled = false;
float animation_speed = 1.f;
float trajectory_radius = 0.75f;
double animation_time = 0.0;
double last_frame_time = 0.0;

bool loadShader(const char* path, std::vector<uint32_t>& out) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file || file.tellg() <= 0 || file.tellg() % 4 != 0) {
        std::cerr << "Cannot read SPIR-V shader: " << path << '\n';
        return false;
    }
    const size_t size = static_cast<size_t>(file.tellg());
    out.resize(size / sizeof(uint32_t));
    file.seekg(0);
    return bool(file.read(reinterpret_cast<char*>(out.data()), std::streamsize(size)));
}

void destroyBuffer(Buffer& buffer) {
    if (buffer.handle != VK_NULL_HANDLE)
        vmaDestroyBuffer(graphics::internal::context.allocator, buffer.handle, buffer.allocation);
    buffer = {};
}

bool createBuffer(Buffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage) {
    const VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocation = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
    };
    VmaAllocationInfo mapped{};
    if (vmaCreateBuffer(graphics::internal::context.allocator, &info, &allocation,
                       &buffer.handle, &buffer.allocation, &mapped) != VK_SUCCESS) {
        resource_error = "Could not allocate a GPU buffer.";
        std::cerr << resource_error << '\n';
        return false;
    }
    buffer.mapped = mapped.pMappedData;
    buffer.size = size;
    return true;
}

void upload(const Buffer& buffer, const void* data, size_t size) {
    if (size == 0) return;
    std::memcpy(buffer.mapped, data, size);
    if (vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, 0, size) != VK_SUCCESS) {
        resource_error = "Could not flush GPU buffer memory.";
        std::cerr << resource_error << '\n';
    }
}

void bindBuffer(VkDescriptorSet set, uint32_t binding, VkDescriptorType type, const Buffer& buffer) {
    const VkDescriptorBufferInfo info = {.buffer = buffer.handle, .offset = 0, .range = buffer.size};
    const VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = type,
        .pBufferInfo = &info,
    };
    vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

bool ensureStorage(Buffer& buffer, size_t count, size_t stride, uint32_t binding) {
    const VkDeviceSize required = std::max(size_t(1), count) * stride;
    if (required <= buffer.size) return true;
    if (required > max_storage_range) {
        resource_error = "The light array exceeds this GPU's storage buffer range.";
        return false;
    }
    Buffer replacement;
    const VkDeviceSize size = std::min(max_storage_range, std::max(required, buffer.size * 2));
    if (!createBuffer(replacement, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) return false;
    bindBuffer(scene_set, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, replacement);
    destroyBuffer(buffer);
    buffer = replacement;
    return true;
}

bool createObjectResources(ObjectResources& object, const Mesh& mesh) {
    object.index_count = static_cast<uint32_t>(mesh.indices.size());
    if (!createBuffer(object.vertices, mesh.vertices.size()*sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) ||
        !createBuffer(object.indices, mesh.indices.size()*sizeof(uint16_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT) ||
        !createBuffer(object.uniform, sizeof(ModelUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) return false;
    upload(object.vertices, mesh.vertices.data(), mesh.vertices.size()*sizeof(Vertex));
    upload(object.indices, mesh.indices.data(), mesh.indices.size()*sizeof(uint16_t));
    bindBuffer(object.descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, object.uniform);
    return true;
}

bool initializeGraphicsPipeline() {
    auto& ctx = graphics::internal::context;
    const VkDescriptorSetLayoutBinding scene_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = scene_bindings,
    };
    if (vkCreateDescriptorSetLayout(ctx.device, &layout_info, nullptr, &scene_layout) != VK_SUCCESS) return false;
    layout_info.bindingCount = 1;
    if (vkCreateDescriptorSetLayout(ctx.device, &layout_info, nullptr, &model_layout) != VK_SUCCESS) return false;
    const VkDescriptorSetLayout layouts[] = {scene_layout, model_layout};
    const VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = layouts,
    };
    if (vkCreatePipelineLayout(ctx.device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) return false;
	std::vector<uint32_t> vert_code;
	std::vector<uint32_t> frag_code;

	if (!loadShader("shaders/triangle.vert.spv", vert_code) ||
	    !loadShader("shaders/triangle.frag.spv", frag_code)) {
		return false;
	}

	VkShaderModuleCreateInfo module_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
	};

	module_info.codeSize = vert_code.size() * sizeof(uint32_t);
	module_info.pCode = reinterpret_cast<const uint32_t*>(vert_code.data());

	VkShaderModule vertex_module;
	if (vkCreateShaderModule(ctx.device, &module_info, nullptr, &vertex_module) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan vertex shader module\n";
		return false;
	}

	module_info.codeSize = frag_code.size() * sizeof(uint32_t);
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
		{
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 3,
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
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(ctx.physical_device, &properties);
    max_storage_range = properties.limits.maxStorageBufferRange;
    const VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
    };
    const VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 3,
        .poolSizeCount = 2,
        .pPoolSizes = sizes,
    };
    if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) return false;
    const VkDescriptorSetLayout layouts[] = {scene_layout, model_layout, model_layout};
    const VkDescriptorSetAllocateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 3,
        .pSetLayouts = layouts,
    };
    VkDescriptorSet sets[3]{};
    if (vkAllocateDescriptorSets(ctx.device, &info, sets) != VK_SUCCESS) return false;
    scene_set = sets[0];
    objects[0].descriptor_set = sets[1];
    objects[1].descriptor_set = sets[2];
    if (!createBuffer(scene_buffer, sizeof(SceneUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) return false;
    bindBuffer(scene_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, scene_buffer);
    if (!ensureStorage(point_buffer, point_lights.size(), sizeof(PointLight), 1) ||
        !ensureStorage(spot_buffer, spot_lights.size(), sizeof(SpotLight), 2)) return false;
    const Mesh mesh = makeTruncatedTetrahedron();
    for (auto& object : objects) {
        if (!createObjectResources(object, mesh)) return false;
    }
    return true;
}

void editLight(PointLight& light) {
    ImGui::DragFloat3("Position", &light.position.x, 0.05f);
    ImGui::ColorEdit3("Color", &light.color.x);
    ImGui::DragFloat("Intensity", &light.color.w, 0.1f, 0.f, 1000.f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
}

void editModel(ModelControls& model) {
    ImGui::DragFloat3("Position", model.position, 0.05f);
    ImGui::DragFloat3("Rotation (deg)", model.rotation, 1.f);
    ImGui::DragFloat3("Scale", model.scale, 0.02f, 0.05f, 5.f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
    ImGui::ColorEdit3("Albedo", model.albedo);
    ImGui::ColorEdit3("Specular color", model.specular);
    ImGui::SliderFloat("Shininess", &model.shininess, 1.f, 256.f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
    ImGui::Checkbox("Procedural vertex colors (Lab 1)", &model.vertex_colors);
}

Vec3 trajectory(double time, float radius, float phase) {
    return {radius * float(std::sin(time + phase)),
            radius * 0.5f * float(std::sin(time * 2.0 + phase)),
            radius * float(std::cos(time + phase))};
}

void updateSceneBuffers(float aspect) {
    const Mat4 projection = projection_index == 0
        ? matPerspective(camera_fov, aspect, 0.1f, 100.f)
        : matOrtho(-ortho_size*aspect, ortho_size*aspect, -ortho_size, ortho_size, 0.1f, 100.f);
    const Mat4 view_projection = matMultiply(matVulkanClipCorrection(),
        matMultiply(projection, cameraView(camera_position, camera_angles)));
    const Mat4 rotation = cameraRotation(camera_angles);
    SceneUniform scene{};
    scene.view_projection = view_projection;
    scene.camera_position = {camera_position[0], camera_position[1], camera_position[2], float(projection_index)};
    scene.camera_forward = {-rotation.m[8], -rotation.m[9], -rotation.m[10], 0.f};
    scene.ambient = {ambient_color[0]*ambient_intensity, ambient_color[1]*ambient_intensity,
                     ambient_color[2]*ambient_intensity, 0.f};
    scene.directional_direction = lightDirection(directional_angles[0], directional_angles[1]);
    scene.directional_color = {directional_color[0], directional_color[1], directional_color[2], directional_intensity};
    if (ensureStorage(point_buffer, point_lights.size(), sizeof(PointLight), 1)) {
        upload(point_buffer, point_lights.data(), point_lights.size()*sizeof(PointLight));
        scene.light_counts[0] = static_cast<uint32_t>(point_lights.size());
    }
    packed_spots.clear();
    for (const auto& spot : spot_lights)
        packed_spots.push_back(makeSpotLight(spot.light, spot.pitch, spot.yaw, spot.inner, spot.outer));
    if (ensureStorage(spot_buffer, packed_spots.size(), sizeof(SpotLight), 2)) {
        upload(spot_buffer, packed_spots.data(), packed_spots.size()*sizeof(SpotLight));
        scene.light_counts[1] = static_cast<uint32_t>(packed_spots.size());
    }
    upload(scene_buffer, &scene, sizeof(scene));
    for (uint32_t i = 0; i < 2; ++i) {
        const ModelControls& m = models[i];
        const float phase = i == 0 ? 0.f : 2.4f;
        const Vec3 offset = subtract(trajectory(animation_time, trajectory_radius, phase),
                                     trajectory(0.0, trajectory_radius, phase));
        const Mat4 model = matMultiply(matTranslate(m.position[0]+offset.x, m.position[1]+offset.y, m.position[2]+offset.z),
            matMultiply(matRotateY(m.rotation[1]+float(animation_time)*(i == 0 ? 45.f : -60.f)),
            matMultiply(matRotateX(m.rotation[0]), matMultiply(matRotateZ(m.rotation[2]),
            matScale(m.scale[0], m.scale[1], m.scale[2])))));
        const ModelUniform data = {model,
            {m.albedo[0], m.albedo[1], m.albedo[2], m.vertex_colors ? 1.f : 0.f},
            {m.specular[0], m.specular[1], m.specular[2], m.shininess}};
        upload(objects[i].uniform, &data, sizeof(data));
    }
}

} // namespace

bool initialize() {
    if (!initializeGraphicsPipeline() || !initializeObjects()) {
        shutdown();
        return false;
    }
    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkDeviceWaitIdle(ctx.device);
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
    vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, scene_layout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, model_layout, nullptr);
    for (auto& object : objects) {
        destroyBuffer(object.vertices);
        destroyBuffer(object.indices);
        destroyBuffer(object.uniform);
    }
    destroyBuffer(scene_buffer);
    destroyBuffer(point_buffer);
    destroyBuffer(spot_buffer);
}

void update(double time) {
    const double delta = last_frame_time == 0.0 ? 0.0 : time-last_frame_time;
    last_frame_time = time;
    if (animation_enabled) animation_time += delta*animation_speed;
    ImGui::SetNextWindowSize(ImVec2(370.f, 650.f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Lab 2: Camera and Lighting");
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.48f);
    if (resource_error) ImGui::TextWrapped("%s", resource_error);
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* projections[] = {"Perspective", "Orthographic"};
        ImGui::Combo("Projection", &projection_index, projections, 2);
        ImGui::DragFloat3("Camera position", camera_position, 0.1f);
        ImGui::DragFloat3("Pitch / yaw / roll", camera_angles, 0.5f);
        ImGui::SliderFloat("FOV (deg)", &camera_fov, 10.f, 120.f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
        if (projection_index == 1)
            ImGui::SliderFloat("Ortho half-height", &ortho_size, 1.f, 20.f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
        if (ImGui::Button("Reset camera")) {
            camera_position[0] = 0.f; camera_position[1] = 4.f; camera_position[2] = 13.f;
            camera_angles[0] = -17.f; camera_angles[1] = camera_angles[2] = 0.f;
            camera_fov = 50.f; ortho_size = 6.f; projection_index = 0;
        }
    }
    if (ImGui::CollapsingHeader("Objects and materials", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (int i = 0; i < 2; ++i) {
            ImGui::PushID(i);
            if (ImGui::TreeNode("Model", "Tetrahedron %d", i+1)) {
                editModel(models[i]);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        ImGui::Checkbox("Animate", &animation_enabled);
        ImGui::SliderFloat("Speed", &animation_speed, 0.f, 3.f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("Trajectory radius", &trajectory_radius, 0.f, 5.f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        if (ImGui::Button("Reset animation")) { animation_time = 0.0; animation_enabled = false; }
    }
    if (ImGui::CollapsingHeader("Ambient and directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient color", ambient_color);
        ImGui::SliderFloat("Ambient intensity", &ambient_intensity, 0.f, 1.f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::ColorEdit3("Directional color", directional_color);
        ImGui::SliderFloat("Directional intensity", &directional_intensity, 0.f, 5.f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::DragFloat2("Light pitch / yaw", directional_angles, 0.5f);
    }
    if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Count: %zu", point_lights.size());
        ImGui::BeginDisabled(point_lights.size() >= max_storage_range/sizeof(PointLight));
        if (ImGui::Button("Add point light")) point_lights.push_back({{0.f, 3.f, 4.f, 0.f}, {1.f, 1.f, 1.f, 20.f}});
        ImGui::EndDisabled();
        for (size_t i = 0; i < point_lights.size();) {
            ImGui::PushID(static_cast<int>(i));
            bool remove = false;
            if (ImGui::TreeNode("Point", "Point %zu", i+1)) {
                editLight(point_lights[i]);
                remove = ImGui::Button("Delete point light");
                ImGui::TreePop();
            }
            ImGui::PopID();
            if (remove) point_lights.erase(point_lights.begin()+i); else ++i;
        }
    }
    if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Count: %zu", spot_lights.size());
        ImGui::BeginDisabled(spot_lights.size() >= max_storage_range/sizeof(SpotLight));
        if (ImGui::Button("Add spot light")) spot_lights.push_back(SpotControls{});
        ImGui::EndDisabled();
        for (size_t i = 0; i < spot_lights.size();) {
            ImGui::PushID(static_cast<int>(i));
            bool remove = false;
            if (ImGui::TreeNode("Spot", "Spot %zu", i+1)) {
                auto& spot = spot_lights[i];
                editLight(spot.light);
                ImGui::DragFloat("Pitch (deg)", &spot.pitch, 0.5f);
                ImGui::DragFloat("Yaw (deg)", &spot.yaw, 0.5f);
                if (ImGui::SliderFloat("Inner half-angle", &spot.inner, 0.f, 89.f, "%.1f deg", ImGuiSliderFlags_AlwaysClamp))
                    spot.outer = std::max(spot.inner, spot.outer);
                if (ImGui::SliderFloat("Outer half-angle", &spot.outer, 0.f, 89.f, "%.1f deg", ImGuiSliderFlags_AlwaysClamp))
                    spot.inner = std::min(spot.inner, spot.outer);
                remove = ImGui::Button("Delete spot light");
                ImGui::TreePop();
            }
            ImGui::PopID();
            if (remove) spot_lights.erase(spot_lights.begin()+i); else ++i;
        }
    }
    ImGui::PopItemWidth();
    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    auto& ctx = graphics::internal::context;
    if (fd.command_buffer == VK_NULL_HANDLE || fd.framebuffer == VK_NULL_HANDLE) return;
    updateSceneBuffers(float(ctx.swapchain_extent.width)/float(ctx.swapchain_extent.height));
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

	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		pipeline_layout, 0, 1, &scene_set, 0, nullptr);
	for (const ObjectResources& object : objects) {
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
								pipeline_layout, 1, 1, &object.descriptor_set, 0, nullptr);

		const VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &object.vertices.handle, &offset);
		vkCmdBindIndexBuffer(fd.command_buffer, object.indices.handle, 0,
							 VK_INDEX_TYPE_UINT16);

		vkCmdDrawIndexed(fd.command_buffer, object.index_count, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(fd.command_buffer);

	if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to record Vulkan command buffer\n";
	}
}

} // namespace application
