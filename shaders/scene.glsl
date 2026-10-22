// These layouts match SceneUniform and ModelUniform in source/scene.hpp.
layout(std140, set = 0, binding = 0) uniform SceneUniform {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 cameraForward;
    vec4 ambient;
    vec4 directionalDirection;
    vec4 directionalColor;
    uvec4 lightCounts;
} scene;

layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec4 albedo;
    vec4 specular;
} material;
