#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

/**
 * Fast ray-AABB intersection test for BVH traversal.
 * Uses the slab method for efficient intersection testing.
 */
__host__ __device__ bool rayBoxIntersect(
    Ray ray,
    glm::vec3 minBounds,
    glm::vec3 maxBounds)
{
    glm::vec3 invDir = 1.0f / ray.direction;
    glm::vec3 t1 = (minBounds - ray.origin) * invDir;
    glm::vec3 t2 = (maxBounds - ray.origin) * invDir;
    
    glm::vec3 tmin = glm::min(t1, t2);
    glm::vec3 tmax = glm::max(t1, t2);
    
    float tenter = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
    float texit = glm::min(glm::min(tmax.x, tmax.y), tmax.z);
    
    return tenter <= texit && texit > 0.0f;
}

/**
 * Ray-triangle intersection using the Möller-Trumbore algorithm.
 * Fast, efficient algorithm that directly computes barycentric coordinates.
 */
__host__ __device__ float rayTriangleIntersect(
    Ray ray,
    Triangle triangle,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    // Use EPSILON from utilities.h (0.00001f)
    
    // Get triangle edges
    glm::vec3 edge1 = triangle.v1 - triangle.v0;
    glm::vec3 edge2 = triangle.v2 - triangle.v0;
    
    // Begin calculating determinant - also used to calculate U parameter
    glm::vec3 h = glm::cross(ray.direction, edge2);
    float a = glm::dot(edge1, h);
    
    // If determinant is near zero, ray lies in plane of triangle
    if (a > -EPSILON && a < EPSILON) {
        return -1.0f;  // No intersection
    }
    
    float f = 1.0f / a;
    glm::vec3 s = ray.origin - triangle.v0;
    float u = f * glm::dot(s, h);
    
    // Check if intersection point lies outside triangle
    if (u < 0.0f || u > 1.0f) {
        return -1.0f;
    }
    
    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(ray.direction, q);
    
    // Check if intersection point lies outside triangle
    if (v < 0.0f || u + v > 1.0f) {
        return -1.0f;
    }
    
    // Calculate t to find where the intersection point is on the line
    float t = f * glm::dot(edge2, q);
    
    if (t > EPSILON) { // Ray intersection
        intersectionPoint = ray.origin + t * ray.direction;
        normal = triangle.normal;
        outside = glm::dot(ray.direction, normal) < 0.0f;
        
        // If ray hits backface, flip normal
        if (!outside) {
            normal = -normal;
        }
        
        return t;
    }
    
    return -1.0f;  // Line intersection but not ray intersection
}
