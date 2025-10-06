#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
#ifdef USE_BVH
    void buildBVH();  // Build BVH acceleration structure
    glm::vec3 computeGeomBounds(int geomIndex, bool getMin); // Helper for BVH construction
    glm::vec3 computeTriangleBounds(int triangleIndex, bool getMin); // Helper for triangle bounds
    
    // Recursive BVH construction functions
    int buildBVHRecursive(int first, int count, int depth);
    void computeBounds(int first, int count, glm::vec3& minBounds, glm::vec3& maxBounds);
    int partition(int first, int count);
#endif // USE_BVH
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    
    // Mesh support
    std::vector<Triangle> triangles;      // Global triangle pool
    std::vector<Mesh> meshes;             // Mesh objects
    
#ifdef USE_BVH
    std::vector<BVHNode> bvhNodes;        // BVH tree nodes
    std::vector<Primitive> primitives;    // Unified primitive list (geoms + triangles)
#endif // USE_BVH
    RenderState state;
};
