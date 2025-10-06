#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
// Note: PI is already defined in utilities.h

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

/**
 * Generate a random point on a unit disk for aperture sampling
 * Uses concentric mapping for uniform distribution
 */
__device__ glm::vec2 concentricSampleDisk(thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float u1 = u01(rng);
    float u2 = u01(rng);
    
    // Map uniform random numbers to [-1,1]
    float sx = 2.0f * u1 - 1.0f;
    float sy = 2.0f * u2 - 1.0f;
    
    // Handle degeneracy at the origin
    if (sx == 0.0f && sy == 0.0f) {
        return glm::vec2(0.0f);
    }
    
    // Apply concentric mapping to point on disk
    float theta, r;
    if (abs(sx) > abs(sy)) {
        r = sx;
        theta = (PI / 4.0f) * (sy / sx);
    } else {
        r = sy;
        theta = (PI / 2.0f) - (PI / 4.0f) * (sx / sy);
    }
    
    return glm::vec2(r * cos(theta), r * sin(theta));
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
// Mesh support
static Triangle* dev_triangles = NULL;

#ifdef USE_BVH
static BVHNode* dev_bvhNodes = NULL;
static Primitive* dev_primitives = NULL;
#endif // USE_BVH

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

#ifdef USE_BVH
    // Initialize BVH acceleration structure
    if (!scene->bvhNodes.empty()) {
        cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BVHNode));
        cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
        
        cudaMalloc(&dev_primitives, scene->primitives.size() * sizeof(Primitive));
        cudaMemcpy(dev_primitives, scene->primitives.data(), scene->primitives.size() * sizeof(Primitive), cudaMemcpyHostToDevice);
    }
#endif // USE_BVH

    // Initialize triangle data
    if (!scene->triangles.empty()) {
        cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
        cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_triangles);
#ifdef USE_BVH
    cudaFree(dev_bvhNodes);
    cudaFree(dev_primitives);
#endif // USE_BVH

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.accumulated_radiance = glm::vec3(0.0f, 0.0f, 0.0f);  // Start with no light accumulated
        segment.current_throughput = glm::vec3(1.0f, 1.0f, 1.0f);   // Start with full throughput

        // Set up random number generator
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        // Add random jitter within the pixel for antialiasing
        float jitterX = (u01(rng) - 0.5f); // Random offset [-0.5, 0.5] pixels
        float jitterY = (u01(rng) - 0.5f); // Random offset [-0.5, 0.5] pixels

        // Calculate ray direction from camera center to pixel (before lens effect)
        glm::vec3 rayDir = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );

        // Calculate focal point (point where ray hits the focal plane)
        glm::vec3 focalPoint = cam.position + rayDir * cam.focalDistance;

        // Apply depth of field if lens radius > 0
        if (cam.lensRadius > 0.0f) {
            // Sample random point on lens aperture
            glm::vec2 diskSample = concentricSampleDisk(rng) * cam.lensRadius;
            
            // Calculate lens offset in world coordinates
            glm::vec3 lensOffset = cam.right * diskSample.x + cam.up * diskSample.y;
            
            // New ray origin is offset from camera center
            segment.ray.origin = cam.position + lensOffset;
            
            // New ray direction goes from lens point to focal point
            segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
        } else {
            // No depth of field - standard pinhole camera
            segment.ray.origin = cam.position;
            segment.ray.direction = rayDir;
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

#ifdef USE_BVH
/**
 * BVH traversal function for accelerated ray-scene intersection
 */
__device__ float traverseBVH(
    BVHNode* bvhNodes,
    Primitive* primitives,
    Geom* geoms,
    Triangle* triangles,
    Ray ray,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    int& hitGeomIndex,
    bool& outside)
{
    if (bvhNodes == NULL) return -1.0f; // No BVH available
    
    int stack[32];  // Traversal stack (32 should be enough for most scenes)
    int stackPtr = 0;
    stack[stackPtr++] = 0;  // Start with root node
    
    float t_min = FLT_MAX;
    hitGeomIndex = -1;
    
    while (stackPtr > 0) {
        int nodeIndex = stack[--stackPtr];
        BVHNode& node = bvhNodes[nodeIndex];
        
        // Test ray against bounding box
        if (!rayBoxIntersect(ray, node.minBounds, node.maxBounds))
            continue;
        
        if (node.leftChild == -1) {  // Leaf node
            // Test against all primitives in this leaf
            for (int i = 0; i < node.primCount; i++) {
                const Primitive& prim = primitives[node.firstPrim + i];
                
                float t;
                glm::vec3 tmp_intersect;
                glm::vec3 tmp_normal;
                bool tmp_outside;
                
                if (prim.type == PRIM_GEOM) {
                    Geom& geom = geoms[prim.index];
                    
                    if (geom.type == CUBE) {
                        t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, tmp_outside);
                    }
                    else if (geom.type == SPHERE) {
                        t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, tmp_outside);
                    }
                    else {
                        continue;
                    }
                    
                    if (t > 0.0f && t < t_min) {
                        t_min = t;
                        hitGeomIndex = prim.index;  // Store the actual geom index
                        intersectionPoint = tmp_intersect;
                        normal = tmp_normal;
                        outside = tmp_outside;
                    }
                }
                else if (prim.type == PRIM_TRIANGLE) {
                    Triangle& triangle = triangles[prim.index];
                    
                    t = rayTriangleIntersect(ray, triangle, tmp_intersect, tmp_normal, tmp_outside);
                    
                    if (t > 0.0f && t < t_min) {
                        t_min = t;
                        hitGeomIndex = -1 - prim.index;  // Store triangle index as negative (offset by 1)
                        intersectionPoint = tmp_intersect;
                        normal = tmp_normal;
                        outside = tmp_outside;
                    }
                }
            }
        } else {  // Internal node
            // Add children to stack (right child first for better traversal order)
            if (node.rightChild != -1)
                stack[stackPtr++] = node.rightChild;
            if (node.leftChild != -1)
                stack[stackPtr++] = node.leftChild;
        }
    }
    
    return (t_min == FLT_MAX) ? -1.0f : t_min;
}
#endif // USE_BVH

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Triangle* triangles,
    int triangles_size,
#ifdef USE_BVH
    BVHNode* bvhNodes,
    Primitive* primitives,
#endif
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        glm::vec3 intersect_point;
        glm::vec3 normal;
        int hit_geom_index = -1;
        bool outside = true;

#ifdef USE_BVH
        // Use BVH traversal for accelerated intersection testing
        float t_min = traverseBVH(
            bvhNodes,
            primitives,
            geoms,
            triangles,
            pathSegment.ray,
            intersect_point,
            normal,
            hit_geom_index,
            outside
        );
#else
        // Naive intersection testing - check all geometries
        float t_min = FLT_MAX;
        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            float t;
            glm::vec3 tmp_intersect;
            glm::vec3 tmp_normal;
            bool tmp_outside;

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }
            else
            {
                continue;
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                outside = tmp_outside;
            }
        }
        
        // Naive intersection testing - check all triangles
        for (int i = 0; i < triangles_size; i++)
        {
            Triangle& triangle = triangles[i];
            
            float t;
            glm::vec3 tmp_intersect;
            glm::vec3 tmp_normal;
            bool tmp_outside;
            
            t = rayTriangleIntersect(pathSegment.ray, triangle, tmp_intersect, tmp_normal, tmp_outside);
            
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = -1 - i;  // Use negative encoding for triangles
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                outside = tmp_outside;
            }
        }
#endif // USE_BVH

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else if (hit_geom_index >= 0)
        {
            // The ray hits a geometry object
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
        else
        {
            // The ray hits a triangle (negative hit_geom_index)
            int triangleIndex = -1 - hit_geom_index;  // Convert back from negative encoding
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = triangles[triangleIndex].materialId;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms,
    int numGeoms,
    Triangle* triangles,
    int numTriangles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // Apply Beer's Law absorption if ray traveled through a refractive material
            // Note: This applies attenuation based on the distance the ray traveled to reach this intersection
            if (material.hasRefractive > 0.0f) {
                // Calculate attenuation using Beer's Law: I = I0 * exp(-k * d)
                float distance = intersection.t;
                glm::vec3 attenuation = glm::exp(-material.absorptionCoefficient * distance);
                pathSegments[idx].current_throughput *= attenuation;
            }

            // If the material indicates that the object was a light, accumulate emission
            if (material.emittance > 0.0f) {
                pathSegments[idx].accumulated_radiance += pathSegments[idx].current_throughput * (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0; // Terminate ray at light source
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                if (pathSegments[idx].remainingBounces <= 0) {
                    pathSegments[idx].remainingBounces = 0; // Mark as terminated but keep accumulated color
                    return;
                }
                pathSegments[idx].remainingBounces--; // Decrement bounces
                glm::vec3 intersect = getPointOnRay(pathSegments[idx].ray, intersection.t);
                
                // Add direct lighting contribution (only for non-emissive materials)
                if (material.emittance <= 0.0f) {
                    glm::vec3 directLighting = sampleDirectLighting(
                        intersect,
                        intersection.surfaceNormal,
                        material,
                        geoms,
                        numGeoms,
                        triangles,
                        numTriangles,
                        materials,
                        rng
                    );
                    
                    // Accumulate direct lighting contribution
                    pathSegments[idx].accumulated_radiance += pathSegments[idx].current_throughput * directLighting;
                }
                
                scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng);
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].accumulated_radiance += pathSegments[idx].current_throughput * glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.accumulated_radiance;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = pixelcount;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_triangles,
            (int)hst_scene->triangles.size(),
#ifdef USE_BVH
            dev_bvhNodes,
            dev_primitives,
#endif
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        // Sort by material ID to improve memory coherence
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths,
            [] __device__ (const ShadeableIntersection& a, const ShadeableIntersection& b) {
                return a.materialId < b.materialId;
            });

        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_triangles,
            hst_scene->triangles.size()
        );
        
        // Partition: move active rays to front, terminated rays to back
        // Using thrust::partition to keep rays with remainingBounces > 0 at front
        PathSegment* partition_point = thrust::stable_partition(thrust::device, dev_paths, dev_path_end,
            [] __device__ (const PathSegment& path) {
                return path.remainingBounces > 0;
            });
        num_paths = partition_point - dev_paths;

        // Check termination conditions
        if (depth >= traceDepth || num_paths == 0) {
            iterationComplete = true;
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    // Process ALL paths (including terminated ones) since partition keeps them all
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
