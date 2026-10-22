#version 450
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec3 vertexColor;
layout(location = 0) out vec4 outColor;

struct PointLight {
    vec4 position;
    vec4 color; // rgb = color, w = intensity
};
struct SpotLight {
    vec4 position;
    vec4 color;
    vec4 direction; // xyz = direction of emitted rays, w = cos(inner half-angle)
    vec4 cone; // x = cos(outer half-angle)
};
layout(std430, set = 0, binding = 1) readonly buffer PointLights { PointLight values[]; } points;
layout(std430, set = 0, binding = 2) readonly buffer SpotLights { SpotLight values[]; } spots;

vec3 safeNormalize(vec3 v) {
    return v * inversesqrt(max(dot(v, v), 1e-12));
}

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo) {
    float diffuse = max(dot(N, L), 0.0);
    if (diffuse <= 0.0) return vec3(0.0); // No highlight on the unlit side.
    vec3 H = safeNormalize(L + V);
    float specular = pow(max(dot(N, H), 0.0), material.specular.w);
    return radiance * (albedo * diffuse + material.specular.rgb * specular);
}

void main() {
    vec3 N = safeNormalize(worldNormal);
    vec3 V = scene.cameraPosition.w > 0.5
        ? -scene.cameraForward.xyz : safeNormalize(scene.cameraPosition.xyz - worldPosition);
    vec3 albedo = material.albedo.rgb * mix(vec3(1.0), vertexColor, material.albedo.w);
    vec3 color = scene.ambient.rgb * albedo;
    color += blinnPhong(N, V, -scene.directionalDirection.xyz,
                       scene.directionalColor.rgb * scene.directionalColor.w, albedo);
    for (uint i = 0u; i < scene.lightCounts.x; ++i) {
        PointLight light = points.values[i];
        vec3 delta = light.position.xyz - worldPosition;
        // Regularize only the singularity at the light's position.
        float attenuation = 1.0 / max(dot(delta, delta), 1e-4);
        color += blinnPhong(N, V, safeNormalize(delta), light.color.rgb * light.color.w * attenuation, albedo);
    }
    for (uint i = 0u; i < scene.lightCounts.y; ++i) {
        SpotLight light = spots.values[i];
        vec3 delta = light.position.xyz - worldPosition;
        vec3 L = safeNormalize(delta);
        float cosAngle = dot(-L, light.direction.xyz);
        float inner = light.direction.w, outer = light.cone.x;
        // Equal angles describe a hard edge; smoothstep requires distinct edges.
        float cone = inner > outer ? smoothstep(outer, inner, cosAngle) : step(outer, cosAngle);
        float attenuation = cone / max(dot(delta, delta), 1e-4);
        color += blinnPhong(N, V, L, light.color.rgb * light.color.w * attenuation, albedo);
    }
    outColor = vec4(color, 1.0);
}
