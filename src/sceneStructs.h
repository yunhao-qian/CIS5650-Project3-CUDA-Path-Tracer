#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"
#include "utilities.h"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    TRIANGLE,
    MESH
};

// BVH (Bounding Volume Hierarchy) structures for acceleration
struct BVHNode
{
    glm::vec3 minBounds;
    glm::vec3 maxBounds;
    int leftChild;   // -1 if leaf node
    int rightChild;  // -1 if leaf node  
    int firstPrim;   // Index of first primitive (for leaf nodes)
    int primCount;   // Number of primitives in this node (for leaf nodes)
};

// Unified primitive reference for BVH
enum PrimitiveType
{
    PRIM_GEOM,      // References geoms array
    PRIM_TRIANGLE   // References triangles array
};

struct Primitive
{
    PrimitiveType type;
    int index;      // Index into either geoms or triangles array
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    glm::vec3 absorptionCoefficient; // For Beer's Law: color attenuation per unit distance
};

// Triangle primitive for mesh support
struct Triangle
{
    glm::vec3 v0, v1, v2;    // Vertices in world space
    glm::vec3 normal;        // Precomputed face normal
    int materialId;          // Material index for this triangle
    
    // Optional: per-vertex normals for smooth shading (future enhancement)
    // glm::vec3 n0, n1, n2;
    
    // Optional: texture coordinates (future enhancement)  
    // glm::vec2 uv0, uv1, uv2;
};

// Mesh object containing multiple triangles
struct Mesh
{
    int firstTriangle;       // Index into global triangle array
    int triangleCount;       // Number of triangles in this mesh
    int materialId;          // Default material for the mesh
    glm::mat4 transform;     // Mesh-level transformation matrix
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    
    // Depth of field parameters
    float lensRadius;    // Aperture radius (0 = no depth of field)
    float focalDistance; // Distance to focal plane
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 accumulated_radiance;  // Accumulated light contribution for this path
    glm::vec3 current_throughput;    // Current path throughput (attenuation factor)
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};

// Light sampling structures for direct lighting
struct LightSample
{
    glm::vec3 position;      // Sample point on light surface
    glm::vec3 normal;        // Surface normal at sample point
    glm::vec3 emission;      // Light color * emittance
    float pdf;               // Probability density of this sample
    float area;              // Light surface area (for normalization)
};

// Light reference for importance sampling
struct LightRef
{
    PrimitiveType type;      // PRIM_GEOM or PRIM_TRIANGLE
    int index;               // Index into geoms or triangles array
    float power;             // Precomputed light power for sampling
};

// Utility functions for direct lighting
__host__ __device__ inline float luminance(const glm::vec3& color) {
    // Standard luminance conversion based on human vision perception
    return 0.299f * color.r + 0.587f * color.g + 0.114f * color.b;
}

__host__ __device__ inline float lightPower(const Material& material, float area) {
    // Calculate total light power for importance sampling
    return luminance(material.color) * material.emittance * area;
}

// Forward declarations for light sampling functions (implemented in interactions.cu)
__host__ __device__ float getGeometryArea(const Geom& geom);
__host__ __device__ float getTriangleArea(const Triangle& tri);
