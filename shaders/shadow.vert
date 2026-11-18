#version 450
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"
layout(location = 0) in vec3 inPosition;
layout(push_constant) uniform ShadowPass { layout(offset = 16) uint lightIndex; } shadowPass;
void main() {
    mat4 lightVP = shadowPass.lightIndex == 0u ? scene.sunViewProjection : scene.spotViewProjection;
    gl_Position = lightVP * material.model * vec4(inPosition, 1.0);
}
