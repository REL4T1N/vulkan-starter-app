#version 450
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec2 textureUV;

void main() {
    vec4 world = material.model * vec4(inPosition, 1.0);
    gl_Position = scene.viewProjection * world;
    worldPosition = world.xyz;
    // Inverse transpose preserves perpendicularity under nonuniform scaling.
    worldNormal = transpose(inverse(mat3(material.model))) * inNormal;
    textureUV = inUV;
}
