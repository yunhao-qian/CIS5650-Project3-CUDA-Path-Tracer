#include "interactions.h"
#include "intersections.h"
#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

/**
 * Generate imperfect specular reflection direction based on GPU Gems 3, Chapter 20
 * Equations 7, 8, and 9 (with correction: chi in text = xi in formulas)
 */
__host__ __device__ glm::vec3 calculateRandomSpecularDirection(
    glm::vec3 incomingDirection,
    glm::vec3 normal,
    float roughness,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    
    // Perfect reflection direction
    glm::vec3 reflectDir = glm::reflect(incomingDirection, normal);
    
    // For perfect specular (roughness = 0), return perfect reflection
    if (roughness <= 0.0f) {
        return reflectDir;
    }
    
    // Generate random variables xi1 and xi2
    float xi1 = u01(rng);
    float xi2 = u01(rng);
    
    // GPU Gems 3, Chapter 20, Equations 7-9
    // Calculate theta and phi for sampling around perfect reflection
    float alpha = roughness * roughness; // Convert roughness to alpha parameter
    float theta = atan(alpha * sqrt(xi1) / sqrt(1.0f - xi1));
    float phi = 2.0f * PI * xi2;
    
    // Convert spherical coordinates to Cartesian in reflection space
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    float cosPhi = cos(phi);
    float sinPhi = sin(phi);
    
    // Build coordinate system around perfect reflection direction
    glm::vec3 w = reflectDir;  // Perfect reflection is our "up" direction
    
    // Find perpendicular vectors (same technique as hemisphere sampling)
    glm::vec3 directionNotReflect;
    if (abs(w.x) < SQRT_OF_ONE_THIRD) {
        directionNotReflect = glm::vec3(1, 0, 0);
    } else if (abs(w.y) < SQRT_OF_ONE_THIRD) {
        directionNotReflect = glm::vec3(0, 1, 0);
    } else {
        directionNotReflect = glm::vec3(0, 0, 1);
    }
    
    glm::vec3 u = glm::normalize(glm::cross(w, directionNotReflect));
    glm::vec3 v = glm::normalize(glm::cross(w, u));
    
    // Sample direction around perfect reflection
    glm::vec3 sampledDir = sinTheta * cosPhi * u + sinTheta * sinPhi * v + cosTheta * w;
    
    return glm::normalize(sampledDir);
}

/**
 * Calculate perfect specular reflection direction
 */
__host__ __device__ glm::vec3 calculatePerfectSpecularDirection(
    glm::vec3 incomingDirection,
    glm::vec3 normal)
{
    return glm::reflect(incomingDirection, normal);
}

/**
 * Calculate Fresnel reflectance using Schlick's approximation
 * Returns the probability of reflection (vs refraction)
 */
__host__ __device__ float calculateFresnelReflectance(
    glm::vec3 incomingDirection,
    glm::vec3 normal,
    float ior1,
    float ior2)
{
    // Ensure we're working with the correct normal direction
    bool entering = glm::dot(incomingDirection, normal) < 0.0f;
    glm::vec3 n = entering ? normal : -normal;
    float eta1 = entering ? ior1 : ior2;
    float eta2 = entering ? ior2 : ior1;
    
    float cosI = glm::abs(glm::dot(incomingDirection, n));
    float sinT2 = (eta1 / eta2) * (eta1 / eta2) * (1.0f - cosI * cosI);
    
    // Total internal reflection
    if (sinT2 >= 1.0f) {
        return 1.0f;
    }
    
    float cosT = sqrt(1.0f - sinT2);
    
    // Schlick's approximation
    float r0 = (eta1 - eta2) / (eta1 + eta2);
    r0 = r0 * r0;
    
    float oneMinusCos = 1.0f - cosI;
    float oneMinusCos2 = oneMinusCos * oneMinusCos;
    float oneMinusCos5 = oneMinusCos2 * oneMinusCos2 * oneMinusCos;
    
    return r0 + (1.0f - r0) * oneMinusCos5;
}

/**
 * Calculate refracted ray direction using Snell's law
 * Returns true if refraction occurs, false for total internal reflection
 */
__host__ __device__ bool calculateRefractionDirection(
    glm::vec3 incomingDirection,
    glm::vec3 normal,
    float ior1,
    float ior2,
    glm::vec3& refractedDirection)
{
    // Determine if ray is entering or exiting the material
    bool entering = glm::dot(incomingDirection, normal) < 0.0f;
    glm::vec3 n = entering ? normal : -normal;
    float eta = entering ? (ior1 / ior2) : (ior2 / ior1);
    
    float cosI = -glm::dot(incomingDirection, n);
    float sinT2 = eta * eta * (1.0f - cosI * cosI);
    
    // Check for total internal reflection
    if (sinT2 >= 1.0f) {
        return false;
    }
    
    float cosT = sqrt(1.0f - sinT2);
    refractedDirection = eta * incomingDirection + (eta * cosI - cosT) * n;
    refractedDirection = glm::normalize(refractedDirection);
    
    return true;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    
    // Determine scattering type based on material properties
    // Use simple probability-based material mixing with energy conservation
    
    float hasReflectiveProbability = m.hasReflective;
    float hasRefractiveProbability = m.hasRefractive;
    
    // Normalize probabilities to ensure energy conservation
    float totalProb = hasReflectiveProbability + hasRefractiveProbability;
    if (totalProb > 1.0f) {
        hasReflectiveProbability /= totalProb;
        hasRefractiveProbability /= totalProb;
    }
    
    float rnd = u01(rng);
    
    // Priority: Refractive > Reflective > Diffuse
    if (hasRefractiveProbability > 0.0f && rnd < hasRefractiveProbability) {
        // Refractive material (glass, water, etc.)
        float ior1 = 1.0f; // Air
        float ior2 = m.indexOfRefraction; // Material IOR
        
        // Calculate Fresnel reflectance to determine reflection vs refraction
        float fresnelReflectance = calculateFresnelReflectance(
            pathSegment.ray.direction, normal, ior1, ior2);
        
        // Use Fresnel reflectance to probabilistically choose reflection or refraction
        thrust::uniform_real_distribution<float> u02(0, 1);
        float fresnelRnd = u02(rng);
        
        if (fresnelRnd < fresnelReflectance) {
            // Fresnel reflection
            pathSegment.ray.direction = calculatePerfectSpecularDirection(
                pathSegment.ray.direction, normal);
            pathSegment.ray.origin = intersect + 0.01f * pathSegment.ray.direction;
        } else {
            // Refraction using Snell's law
            glm::vec3 refractedDir;
            bool canRefract = calculateRefractionDirection(
                pathSegment.ray.direction, normal, ior1, ior2, refractedDir);
            
            if (canRefract) {
                pathSegment.ray.direction = refractedDir;
                // Offset ray origin in the direction of refraction
                pathSegment.ray.origin = intersect + 0.01f * refractedDir;
            } else {
                // Total internal reflection
                pathSegment.ray.direction = calculatePerfectSpecularDirection(
                    pathSegment.ray.direction, normal);
                pathSegment.ray.origin = intersect + 0.01f * pathSegment.ray.direction;
            }
        }
        
        // Note: Beer's law absorption is already applied in pathtrace.cu
        // Don't apply material color absorption here to avoid double attenuation
    }
    else if (hasReflectiveProbability > 0.0f && rnd < (hasRefractiveProbability + hasReflectiveProbability)) {
        // Specular reflection with roughness
        float roughness = 1.0f - m.specular.exponent / 100.0f; // Convert exponent to roughness
        roughness = glm::clamp(roughness, 0.0f, 1.0f);
        
        pathSegment.ray.direction = calculateRandomSpecularDirection(
            pathSegment.ray.direction, normal, roughness, rng);
        pathSegment.ray.origin = intersect + 0.01f * pathSegment.ray.direction;
        
        // Attenuate by specular color
        pathSegment.current_throughput *= m.specular.color;
    }
    else {
        // Diffuse scattering (Lambert BRDF)
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect + 0.01f * pathSegment.ray.direction;
        
        // Attenuate by diffuse color
        pathSegment.current_throughput *= m.color;
    }
    
    // Apply Russian roulette termination to prevent energy loss for long paths
    if (pathSegment.remainingBounces < 5) { // Only apply after a few bounces
        float throughputMagnitude = glm::length(pathSegment.current_throughput);
        float survivalProbability = glm::min(0.95f, throughputMagnitude);
        
        thrust::uniform_real_distribution<float> roulette(0, 1);
        if (roulette(rng) > survivalProbability) {
            // Terminate path
            pathSegment.remainingBounces = 0;
            return;
        }
        // Boost throughput to maintain energy conservation
        pathSegment.current_throughput /= survivalProbability;
    }
    
    // Note: remainingBounces is already decremented in shadeMaterial, don't double-decrement
}

// Implementation of light sampling functions from sceneStructs.h

// Area calculation for different geometry types
__host__ __device__ float getGeometryArea(const Geom& geom) {
    if (geom.type == SPHERE) {
        // Sphere surface area = 4*pi*r^2
        // For non-uniform scaling, use average radius
        float avgRadius = (geom.scale.x + geom.scale.y + geom.scale.z) / 3.0f;
        return 4.0f * PI * avgRadius * avgRadius;
    } else if (geom.type == CUBE) {
        // Cube surface area with non-uniform scaling
        // Surface area = 2*(xy + xz + yz) where x,y,z are the scale factors
        float x = geom.scale.x;
        float y = geom.scale.y; 
        float z = geom.scale.z;
        return 2.0f * (x*y + x*z + y*z);
    }
    return 0.0f; // Unknown geometry type
}

// Calculate triangle area using cross product
__host__ __device__ float getTriangleArea(const Triangle& tri) {
    glm::vec3 edge1 = tri.v1 - tri.v0;
    glm::vec3 edge2 = tri.v2 - tri.v0;
    return 0.5f * glm::length(glm::cross(edge1, edge2));
}

// Sample a point on a sphere uniformly
__host__ __device__ LightSample sampleSphereLight(const Geom& sphere, const Material& material, 
    thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    
    // Uniform sphere sampling
    float u1 = u01(rng);
    float u2 = u01(rng);
    
    float theta = 2.0f * PI * u1;     // Azimuthal angle
    float phi = acos(1.0f - 2.0f * u2); // Polar angle (uniform on sphere)
    
    // Local sphere coordinates
    glm::vec3 localDir = glm::vec3(
        sin(phi) * cos(theta),
        sin(phi) * sin(theta), 
        cos(phi)
    );
    
    // Transform to world coordinates
    glm::vec3 worldPos = glm::vec3(sphere.transform * glm::vec4(localDir * sphere.scale.x, 1.0f));
    glm::vec3 worldNormal = glm::normalize(glm::vec3(sphere.invTranspose * glm::vec4(localDir, 0.0f)));
    
    float area = getGeometryArea(sphere);
    float pdf = 1.0f / glm::max(area, 0.0001f); // Uniform sampling PDF with protection
    
    LightSample sample;
    sample.position = worldPos;
    sample.normal = worldNormal;
    // Emission as radiance (conventional): brightness per unit area
    // This matches how indirect lighting treats emittance
    sample.emission = material.color * material.emittance;
    sample.pdf = pdf;
    sample.area = area;
    
    return sample;
}

// Sample a point on a cube face uniformly with area-weighted face selection
__host__ __device__ LightSample sampleCubeLight(const Geom& cube, const Material& material,
    thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    
    // Calculate face areas for proper area-weighted sampling
    float x = cube.scale.x;
    float y = cube.scale.y;
    float z = cube.scale.z;
    
    float areaXY = x * y; // +Z and -Z faces
    float areaXZ = x * z; // +Y and -Y faces  
    float areaYZ = y * z; // +X and -X faces
    
    float totalArea = 2.0f * (areaXY + areaXZ + areaYZ);
    
    // Cumulative probabilities for area-weighted face selection
    float probXZ = 2.0f * areaXZ / totalArea;  // faces 2,3 (+Y,-Y) 
    float probYZ = 2.0f * areaYZ / totalArea;  // faces 0,1 (+X,-X)

    float rnd = u01(rng);
    int face;
    
    if (rnd < probYZ) {
        // Sample +X or -X face
        face = (u01(rng) < 0.5f) ? 0 : 1;
    } else if (rnd < probYZ + probXZ) {
        // Sample +Y or -Y face
        face = (u01(rng) < 0.5f) ? 2 : 3;
    } else {
        // Sample +Z or -Z face
        face = (u01(rng) < 0.5f) ? 4 : 5;
    }
    
    // Random point within face
    float u = u01(rng) - 0.5f; // [-0.5, 0.5]
    float v = u01(rng) - 0.5f; // [-0.5, 0.5]
    
    glm::vec3 localPos;
    glm::vec3 localNormal;
    
    // Define cube faces in local coordinates
    switch (face) {
        case 0: localPos = glm::vec3(0.5f, u, v); localNormal = glm::vec3(1, 0, 0); break;  // +X
        case 1: localPos = glm::vec3(-0.5f, u, v); localNormal = glm::vec3(-1, 0, 0); break; // -X
        case 2: localPos = glm::vec3(u, 0.5f, v); localNormal = glm::vec3(0, 1, 0); break;  // +Y
        case 3: localPos = glm::vec3(u, -0.5f, v); localNormal = glm::vec3(0, -1, 0); break; // -Y
        case 4: localPos = glm::vec3(u, v, 0.5f); localNormal = glm::vec3(0, 0, 1); break;  // +Z
        case 5: localPos = glm::vec3(u, v, -0.5f); localNormal = glm::vec3(0, 0, -1); break; // -Z
    }
    
    // Transform to world coordinates
    glm::vec3 worldPos = glm::vec3(cube.transform * glm::vec4(localPos * cube.scale, 1.0f));
    glm::vec3 worldNormal = glm::normalize(glm::vec3(cube.invTranspose * glm::vec4(localNormal, 0.0f)));
    
    float area = getGeometryArea(cube);
    float pdf = 1.0f / glm::max(area, 0.0001f); // Uniform sampling PDF with protection
    
    LightSample sample;
    sample.position = worldPos;
    sample.normal = worldNormal;
    // Emission as radiance (conventional): brightness per unit area
    sample.emission = material.color * material.emittance;
    sample.pdf = pdf;
    sample.area = area;
    
    return sample;
}

// Sample a point on a triangle uniformly using barycentric coordinates
__host__ __device__ LightSample sampleTriangleLight(const Triangle& triangle, const Material& material,
    thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    
    // Uniform triangle sampling using barycentric coordinates
    float u1 = u01(rng);
    float u2 = u01(rng);
    
    // Transform to ensure uniform distribution over triangle
    float sqrtU1 = sqrt(u1);
    float alpha = 1.0f - sqrtU1;
    float beta = u2 * sqrtU1;
    float gamma = 1.0f - alpha - beta;
    
    // Barycentric interpolation
    glm::vec3 worldPos = alpha * triangle.v0 + beta * triangle.v1 + gamma * triangle.v2;
    
    float area = getTriangleArea(triangle);
    float pdf = 1.0f / glm::max(area, 0.0001f); // Uniform sampling PDF with protection
    
    LightSample sample;
    sample.position = worldPos;
    sample.normal = triangle.normal;
    // Emission as radiance (conventional): brightness per unit area  
    sample.emission = material.color * material.emittance;
    sample.pdf = pdf;
    sample.area = area;    return sample;
}

// Simple direct lighting with random light selection and proper PDF normalization
__host__ __device__ glm::vec3 sampleDirectLighting(
    glm::vec3 hitPoint,
    glm::vec3 normal,
    Material hitMaterial,
    Geom* geoms,
    int numGeoms,
    Triangle* triangles,
    int numTriangles,
    Material* materials,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    
    glm::vec3 directContribution(0.0f);
    
    // Count emissive lights first
    int numLights = 0;
    for (int i = 0; i < numGeoms; i++) {
        Material material = materials[geoms[i].materialid];
        if (material.emittance > 0.0f) {
            numLights++;
        }
    }
    
    if (numLights == 0) {
        return directContribution; // No lights in scene
    }
    
    // Randomly select one light to sample (uniform selection)
    int selectedLight = (int)(u01(rng) * numLights);
    int lightIndex = 0;
    
    for (int i = 0; i < numGeoms; i++) {
        Material material = materials[geoms[i].materialid];
        if (material.emittance > 0.0f) {
            if (lightIndex == selectedLight) {
                // Sample this light
                LightSample lightSample;
                
                if (geoms[i].type == SPHERE) {
                    lightSample = sampleSphereLight(geoms[i], material, rng);
                } else if (geoms[i].type == CUBE) {
                    lightSample = sampleCubeLight(geoms[i], material, rng);
                } else {
                    return directContribution; // Skip unknown geometry types
                }
                
                // Calculate light direction and distance
                glm::vec3 lightDir = lightSample.position - hitPoint;
                float lightDistance = glm::length(lightDir);
                lightDir /= lightDistance; // Normalize
                
                // Check if light is above surface (avoid self-intersection)
                float NdotL = glm::dot(normal, lightDir);
                if (NdotL > 0.0f) {
                    // Also check if the light surface is facing toward the hit point
                    float cosTheta_light = glm::dot(lightSample.normal, -lightDir);
                    if (cosTheta_light <= 0.0f) {
                        // Light surface is facing away from hit point, skip this sample
                        break;
                    }
                    
                    // Robust shadow ray offset to avoid self-intersection near edges
                    // Use larger epsilon for surfaces near edges and scale with scene
                    float baseEpsilon = 0.001f;
                    float adaptiveEpsilon = glm::max(baseEpsilon, lightDistance * 0.0001f);
                    
                    // For points near edges, use a more conservative offset
                    // Move along normal and slightly toward light to avoid edge cases
                    glm::vec3 shadowRayOrigin = hitPoint + normal * adaptiveEpsilon;
                    
                    // Additional offset if we're very close to the light to avoid precision issues
                    if (lightDistance < 0.1f) {
                        shadowRayOrigin += lightDir * adaptiveEpsilon;
                    }
                    Ray shadowRay;
                    shadowRay.origin = shadowRayOrigin;
                    shadowRay.direction = lightDir;
                    
                    // Check if path to light is clear
                    bool inShadow = false;
                    for (int j = 0; j < numGeoms; j++) {
                        if (j == i) continue; // Skip the light itself
                        
                        glm::vec3 intersectionPoint;
                        glm::vec3 normal_shadow;
                        bool outside;
                        
                        float shadowT = -1.0f;
                        if (geoms[j].type == SPHERE) {
                            shadowT = sphereIntersectionTest(geoms[j], shadowRay, intersectionPoint, normal_shadow, outside);
                        } else if (geoms[j].type == CUBE) {
                            shadowT = boxIntersectionTest(geoms[j], shadowRay, intersectionPoint, normal_shadow, outside);
                        }
                        
                        if (shadowT > adaptiveEpsilon && shadowT < lightDistance - adaptiveEpsilon) {
                            inShadow = true;
                            break;
                        }
                    }
                    
                    if (!inShadow) {
                        // Lambert BRDF for diffuse materials
                        glm::vec3 brdf = hitMaterial.color / PI;
                        
                        // Calculate geometric term for area light
                        float cosTheta_light = glm::dot(lightSample.normal, -lightDir);
                        // Protect against division by zero for very close lights
                        float distanceSquared = glm::max(lightDistance * lightDistance, 0.0001f);
                        
                        // Correct Monte Carlo direct lighting formula with radiance emission:
                        // Contribution = BRDF * Le * cos(theta_surface) * cos(theta_light) / distance^2
                        glm::vec3 lightContribution = brdf * lightSample.emission * NdotL * cosTheta_light / distanceSquared;
                        
                        // Proper MIS weighting to avoid double-counting with indirect lighting
                        // MIS weight = (light_pdf) / (light_pdf + bsdf_pdf)
                        // For simplicity, approximate BSDF PDF as 1/PI for Lambertian surfaces
                        float lightPdf = lightSample.pdf / float(numLights); // Adjusted for light selection
                        float bsdfPdf = 1.0f / PI; // Lambertian BSDF PDF
                        float misWeight = lightPdf / (lightPdf + bsdfPdf);
                        
                        // Apply MIS weight to prevent double-counting
                        directContribution += lightContribution * float(numLights) * misWeight;
                    }
                }
                break;
            }
            lightIndex++;
        }
    }
    
    return directContribution;
}
