#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace application::scene {

constexpr float kPi = 3.14159265358979323846f;

// Матрицы хранятся в column-major порядке: m[column * 4 + row],
struct Vec3 {
	float x, y, z;
};

struct Mat4 {
	// Column-major storage: m[column * 4 + row]
	float m[16];
};

inline Mat4 matIdentity() {
	return { .m = {
		1.f, 0.f, 0.f, 0.f,
		0.f, 1.f, 0.f, 0.f,
		0.f, 0.f, 1.f, 0.f,
		0.f, 0.f, 0.f, 1.f,
	}};
}

// Умножение матриц: r = a * b (применение: сначала b, потом a)
inline Mat4 matMultiply(const Mat4& a, const Mat4& b) {
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

inline Mat4 matTranslate(float x, float y, float z) {
	Mat4 r = matIdentity();
	r.m[12] = x;
	r.m[13] = y;
	r.m[14] = z;
	return r;
}

inline Mat4 matScale(float x, float y, float z) {
	Mat4 r = matIdentity();
	r.m[0] = x;
	r.m[5] = y;
	r.m[10] = z;
	return r;
}

inline Mat4 matRotateX(float degrees) {
	const float a = degrees * kPi / 180.f;
	const float c = std::cos(a), s = std::sin(a);
	return { .m = {
		1.f, 0.f, 0.f, 0.f,
		0.f, c, s, 0.f,
		0.f, -s, c, 0.f,
		0.f, 0.f, 0.f, 1.f,
	}};
}

inline Mat4 matRotateY(float degrees) {
	const float a = degrees * kPi / 180.f;
	const float c = std::cos(a), s = std::sin(a);
	return { .m = {
		c, 0.f, -s, 0.f,
		0.f, 1.f, 0.f, 0.f,
		s, 0.f, c, 0.f,
		0.f, 0.f, 0.f, 1.f,
	}};
}

inline Mat4 matRotateZ(float degrees) {
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
inline Mat4 matPerspective(float fovYDegrees, float aspect, float zNear, float zFar) {
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
inline Mat4 matOrtho(float left, float right, float bottom, float top, float zNear, float zFar) {
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
inline Mat4 matLookAt(Vec3 eye, Vec3 center, Vec3 up) {
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
inline Mat4 matVulkanClipCorrection() {
	return { .m = {
		1.f, 0.f, 0.f, 0.f,
		0.f, -1.f, 0.f, 0.f,
		0.f, 0.f, 0.5f, 0.f,
		0.f, 0.f, 0.5f, 1.f,
	}};
}


inline Vec3 subtract(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
inline float dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline Vec3 normalized(Vec3 v) {
    const float length = std::sqrt(dot(v, v));
    return length > 0.f ? Vec3{v.x/length, v.y/length, v.z/length} : Vec3{};
}

struct Vertex { Vec3 position, color, normal; };

inline constexpr Vec3 kTruncatedTetrahedronPositions[12] = {
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


struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;
};

inline Mesh makeTruncatedTetrahedron() {
    // Each face owns its vertices: adjacent faces must have different normals.
    const std::vector<std::vector<uint16_t>> faces = {
        {0,1,2}, {3,5,4}, {6,7,8}, {9,10,11},
        {0,3,4,7,6,1}, {0,2,9,10,5,3},
        {1,6,8,11,9,2}, {4,5,10,11,8,7},
    };
    Mesh mesh;
    for (auto face : faces) {
        const Vec3 a = kTruncatedTetrahedronPositions[face[0]];
        Vec3 normal = normalized(cross(
            subtract(kTruncatedTetrahedronPositions[face[1]], a),
            subtract(kTruncatedTetrahedronPositions[face[2]], a)));
        // The convex solid is centered at the origin. Ensure outward winding.
        if (dot(normal, a) < 0.f) {
            normal = {-normal.x, -normal.y, -normal.z};
            std::reverse(face.begin(), face.end());
        }
        const uint16_t base = static_cast<uint16_t>(mesh.vertices.size());
        for (uint16_t index : face) {
            const Vec3 p = kTruncatedTetrahedronPositions[index];
            mesh.vertices.push_back({p, {0.5f+p.x/6.f, 0.5f+p.y/6.f, 0.5f+p.z/6.f}, normal});
        }
        for (uint16_t i = 1; i+1 < face.size(); ++i) {
            mesh.indices.insert(mesh.indices.end(), {base, uint16_t(base+i), uint16_t(base+i+1)});
        }
    }
    return mesh;
}

// All GPU records use 16-byte fields; these offsets match std140/std430 in GLSL.
struct alignas(16) Float4 { float x, y, z, w; };
struct alignas(16) SceneUniform {
    Mat4 view_projection;
    Float4 camera_position; // w = 1 for orthographic projection
    Float4 camera_forward;
    Float4 ambient;
    Float4 directional_direction;
    Float4 directional_color; // rgb = color, w = intensity
    uint32_t light_counts[4]; // point, spot, padding, padding
};
struct alignas(16) ModelUniform {
    Mat4 model;
    Float4 albedo; // w = use procedural vertex color
    Float4 specular; // w = shininess
};
struct alignas(16) PointLight {
    Float4 position;
    Float4 color; // w = intensity
};
struct alignas(16) SpotLight {
    Float4 position;
    Float4 color;
    Float4 direction; // w = cos(inner half-angle)
    Float4 cone; // x = cos(outer half-angle)
};
// static_assert(sizeof(SceneUniform) == 160 && offsetof(SceneUniform, light_counts) == 144);
// static_assert(sizeof(ModelUniform) == 96 && offsetof(ModelUniform, albedo) == 64);
// static_assert(sizeof(PointLight) == 32 && offsetof(PointLight, color) == 16);
// static_assert(sizeof(SpotLight) == 64 && offsetof(SpotLight, direction) == 32);

inline Mat4 cameraRotation(const float angles[3]) {
    return matMultiply(matRotateY(angles[1]), matMultiply(matRotateX(angles[0]), matRotateZ(angles[2])));
}
inline Mat4 cameraView(const float position[3], const float angles[3]) {
    return matMultiply(matRotateZ(-angles[2]), matMultiply(matRotateX(-angles[0]),
        matMultiply(matRotateY(-angles[1]), matTranslate(-position[0], -position[1], -position[2]))));
}
inline Float4 lightDirection(float pitch, float yaw, float w = 0.f) {
    const float p = pitch*kPi/180.f, y = yaw*kPi/180.f;
    return {-std::sin(y)*std::cos(p), std::sin(p), -std::cos(y)*std::cos(p), w};
}
inline SpotLight makeSpotLight(PointLight light, float pitch, float yaw, float inner, float outer) {
    inner = std::clamp(inner, 0.f, 89.f);
    outer = std::clamp(outer, inner, 89.f);
    return {light.position, light.color, lightDirection(pitch, yaw, std::cos(inner*kPi/180.f)),
        {std::cos(outer*kPi/180.f), 0.f, 0.f, 0.f}};
}

} // namespace application::scene
