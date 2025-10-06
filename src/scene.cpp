#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            
            // Load specular properties
            if (p.contains("SPECULAR_RGB")) {
                const auto& specCol = p["SPECULAR_RGB"];
                newMaterial.specular.color = glm::vec3(specCol[0], specCol[1], specCol[2]);
            } else {
                newMaterial.specular.color = glm::vec3(1.0f); // Default white specular
            }
            
            if (p.contains("ROUGHNESS")) {
                // Convert roughness to specular exponent (higher exponent = lower roughness)
                float roughness = glm::clamp(static_cast<float>(p["ROUGHNESS"]), 0.0f, 1.0f);
                newMaterial.specular.exponent = (1.0f - roughness) * 100.0f;
            } else if (p.contains("SPECULAR_EXP")) {
                newMaterial.specular.exponent = glm::max(0.0f, static_cast<float>(p["SPECULAR_EXP"]));
            } else {
                newMaterial.specular.exponent = 50.0f; // Default moderate shininess
            }
            
            newMaterial.hasReflective = 1.0f; // Mark as reflective material
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            
            if (p.contains("IOR")) {
                newMaterial.indexOfRefraction = glm::max(1.0f, static_cast<float>(p["IOR"]));
            } else {
                newMaterial.indexOfRefraction = 1.5f; // Default glass IOR
            }
            
            // Load absorption coefficient for Beer's Law
            if (p.contains("ABSORPTION")) {
                const auto& absorption = p["ABSORPTION"];
                // Ensure non-negative absorption coefficients
                newMaterial.absorptionCoefficient = glm::vec3(
                    glm::max(0.0f, static_cast<float>(absorption[0])),
                    glm::max(0.0f, static_cast<float>(absorption[1])),
                    glm::max(0.0f, static_cast<float>(absorption[2]))
                );
            } else if (p.contains("ABSORPTION_DISTANCE")) {
                // Alternative: specify distance for 50% absorption (more intuitive)
                float distance = glm::max(0.001f, static_cast<float>(p["ABSORPTION_DISTANCE"]));
                // Calculate coefficient such that 50% light remains after specified distance
                // Using Beer's Law: I = I0 * exp(-k * d), for 50% transmission: 0.5 = exp(-k * d)
                float coefficient = -log(0.5f) / distance;
                newMaterial.absorptionCoefficient = glm::vec3(coefficient);
            } else {
                // Default: very low absorption (clear glass)
                newMaterial.absorptionCoefficient = glm::vec3(0.1f, 0.1f, 0.1f);
            }
            
            newMaterial.hasRefractive = 1.0f; // Mark as refractive material
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    // Load mesh data if present
    if (data.contains("Meshes"))
    {
        const auto& meshesData = data["Meshes"];
        cout << "Loading " << meshesData.size() << " mesh(es)..." << endl;
        
        for (const auto& meshData : meshesData)
        {
            // Parse vertices array
            const auto& verticesArray = meshData["vertices"];
            std::vector<glm::vec3> vertices;
            for (size_t i = 0; i < verticesArray.size(); i += 3)
            {
                vertices.emplace_back(verticesArray[i], verticesArray[i+1], verticesArray[i+2]);
            }
            
            // Parse triangles array (indices into vertices)
            const auto& trianglesArray = meshData["triangles"];
            std::vector<glm::ivec3> triangleIndices;
            for (size_t i = 0; i < trianglesArray.size(); i += 3)
            {
                triangleIndices.emplace_back(trianglesArray[i], trianglesArray[i+1], trianglesArray[i+2]);
            }
            
            // Parse normals array (optional, per-triangle)
            std::vector<glm::vec3> normals;
            if (meshData.contains("normals"))
            {
                const auto& normalsArray = meshData["normals"];
                for (size_t i = 0; i < normalsArray.size(); i += 3)
                {
                    normals.emplace_back(normalsArray[i], normalsArray[i+1], normalsArray[i+2]);
                }
            }
            
            // Get material for this mesh
            uint32_t materialId = MatNameToID[meshData["MATERIAL"]];
            
            // Create Triangle objects from the mesh data
            int firstTriangleIndex = triangles.size();
            for (size_t i = 0; i < triangleIndices.size(); ++i)
            {
                Triangle tri;
                const glm::ivec3& indices = triangleIndices[i];
                tri.v0 = vertices[indices.x];
                tri.v1 = vertices[indices.y]; 
                tri.v2 = vertices[indices.z];
                tri.materialId = materialId;
                
                // Use provided normal or compute from vertices
                if (i < normals.size())
                {
                    tri.normal = normals[i];
                }
                else
                {
                    // Compute normal from triangle vertices
                    glm::vec3 edge1 = tri.v1 - tri.v0;
                    glm::vec3 edge2 = tri.v2 - tri.v0;
                    tri.normal = glm::normalize(glm::cross(edge1, edge2));
                }
                
                triangles.push_back(tri);
            }
            
            // Create Mesh object
            Mesh mesh;
            mesh.firstTriangle = firstTriangleIndex;
            mesh.triangleCount = triangleIndices.size();
            mesh.materialId = materialId;
            
            // Parse transform data (optional) and build matrices
            glm::vec3 translation(0.0f);
            glm::vec3 rotation(0.0f);
            glm::vec3 scale(1.0f);
            
            if (meshData.contains("TRANS"))
            {
                const auto& trans = meshData["TRANS"];
                translation = glm::vec3(trans[0], trans[1], trans[2]);
            }
            
            if (meshData.contains("ROTAT"))
            {
                const auto& rotat = meshData["ROTAT"];
                rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            }
            
            if (meshData.contains("SCALE"))
            {
                const auto& scale_data = meshData["SCALE"];
                scale = glm::vec3(scale_data[0], scale_data[1], scale_data[2]);
            }
            
            // Build transformation matrices
            mesh.transform = utilityCore::buildTransformationMatrix(translation, rotation, scale);
            mesh.inverseTransform = glm::inverse(mesh.transform);
            mesh.invTranspose = glm::inverseTranspose(mesh.transform);
            
            meshes.push_back(mesh);
            
            cout << "  Loaded mesh with " << triangleIndices.size() << " triangles from " 
                 << vertices.size() << " vertices" << endl;
        }
        
        cout << "Total triangles loaded: " << triangles.size() << endl;
    }

    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "mesh")
        {
            newGeom.type = MESH;
        }
        else if (type == "triangle")
        {
            newGeom.type = TRIANGLE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    // Load depth of field parameters (optional, default to no depth of field)
    camera.lensRadius = 0.0f;  // Default: no depth of field
    camera.focalDistance = glm::length(camera.lookAt - camera.position);  // Default: focus at lookAt point
    
    if (cameraData.contains("LENS_RADIUS")) {
        camera.lensRadius = cameraData["LENS_RADIUS"];
    }
    if (cameraData.contains("FOCAL_DISTANCE")) {
        camera.focalDistance = cameraData["FOCAL_DISTANCE"];
    }

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
    
#ifdef USE_BVH
    // Build BVH acceleration structure
    buildBVH();
    cout << "Built BVH with " << bvhNodes.size() << " nodes" << endl;
#else
    cout << "Using naive intersection (BVH disabled)" << endl;
#endif
}

#ifdef USE_BVH
glm::vec3 Scene::computeGeomBounds(int geomIndex, bool getMin)
{
    const Geom& geom = geoms[geomIndex];
    glm::vec3 bounds;
    
    if (geom.type == SPHERE) {
        // For sphere: center +/- radius in all directions
        glm::vec3 center = glm::vec3(geom.transform[3]);
        float radius = glm::length(glm::vec3(geom.transform[0])); // Approximate radius from scale
        bounds = getMin ? (center - glm::vec3(radius)) : (center + glm::vec3(radius));
    }
    else if (geom.type == CUBE) {
        // For cube: transform the 8 corners and find min/max
        glm::vec3 corners[8] = {
            glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.5f, -0.5f, -0.5f),
            glm::vec3(-0.5f,  0.5f, -0.5f), glm::vec3(0.5f,  0.5f, -0.5f),
            glm::vec3(-0.5f, -0.5f,  0.5f), glm::vec3(0.5f, -0.5f,  0.5f),
            glm::vec3(-0.5f,  0.5f,  0.5f), glm::vec3(0.5f,  0.5f,  0.5f)
        };
        
        bounds = getMin ? glm::vec3(FLT_MAX) : glm::vec3(-FLT_MAX);
        for (int i = 0; i < 8; i++) {
            glm::vec3 worldCorner = glm::vec3(geom.transform * glm::vec4(corners[i], 1.0f));
            if (getMin) {
                bounds = glm::min(bounds, worldCorner);
            } else {
                bounds = glm::max(bounds, worldCorner);
            }
        }
    }
    
    return bounds;
}

glm::vec3 Scene::computeTriangleBounds(int triangleIndex, bool getMin)
{
    const Triangle& triangle = triangles[triangleIndex];
    
    if (getMin) {
        return glm::min(glm::min(triangle.v0, triangle.v1), triangle.v2);
    } else {
        return glm::max(glm::max(triangle.v0, triangle.v1), triangle.v2);
    }
}
#endif // USE_BVH

#ifdef USE_BVH
void Scene::buildBVH()
{
    // Create unified primitive list
    primitives.clear();
    
    // Add all geometry primitives
    for (int i = 0; i < geoms.size(); i++) {
        Primitive prim;
        prim.type = PRIM_GEOM;
        prim.index = i;
        primitives.push_back(prim);
    }
    
    // Add all triangle primitives 
    for (int i = 0; i < triangles.size(); i++) {
        Primitive prim;
        prim.type = PRIM_TRIANGLE;
        prim.index = i;
        primitives.push_back(prim);
    }
    
    if (primitives.empty()) return;
    
    // Reserve space for BVH nodes (worst case: 2*n-1 nodes for n primitives)
    bvhNodes.clear();
    bvhNodes.reserve(2 * primitives.size());
    
    // Build BVH recursively starting from root
    buildBVHRecursive(0, primitives.size(), 0);
    
    std::cout << "Built BVH with " << bvhNodes.size() << " nodes for " << primitives.size() 
              << " primitives (" << geoms.size() << " geoms + " << triangles.size() << " triangles)" << std::endl;
}

int Scene::buildBVHRecursive(int first, int count, int depth) {
    // Add node to array first (pre-order) to ensure root is at index 0
    int nodeIndex = bvhNodes.size();
    bvhNodes.emplace_back(); // Add empty node, we'll fill it in
    BVHNode& node = bvhNodes[nodeIndex];
    
    // Compute bounding box for primitives in range [first, first+count)
    computeBounds(first, count, node.minBounds, node.maxBounds);
    
    // Stopping criteria: create leaf if few primitives or max depth reached
    if (count <= 4 || depth >= 20) {
        // Create leaf node
        node.leftChild = -1;
        node.rightChild = -1;
        node.firstPrim = first;
        node.primCount = count;
    } else {
        // Create internal node - subdivide primitives
        int mid = partition(first, count);
        
        node.firstPrim = 0;  // Internal nodes don't store primitives
        node.primCount = 0;
        
        // Recursively build children
        node.leftChild = buildBVHRecursive(first, mid - first, depth + 1);
        node.rightChild = buildBVHRecursive(mid, first + count - mid, depth + 1);
    }
    
    return nodeIndex;
}

void Scene::computeBounds(int first, int count, glm::vec3& minBounds, glm::vec3& maxBounds) {
    minBounds = glm::vec3(FLT_MAX);
    maxBounds = glm::vec3(-FLT_MAX);
    
    for (int i = 0; i < count; i++) {
        const Primitive& prim = primitives[first + i];
        
        glm::vec3 primMin, primMax;
        
        if (prim.type == PRIM_GEOM) {
            primMin = computeGeomBounds(prim.index, true);
            primMax = computeGeomBounds(prim.index, false);
        } else if (prim.type == PRIM_TRIANGLE) {
            primMin = computeTriangleBounds(prim.index, true);
            primMax = computeTriangleBounds(prim.index, false);
        }
        
        minBounds = glm::min(minBounds, primMin);
        maxBounds = glm::max(maxBounds, primMax);
    }
}

int Scene::partition(int first, int count) {
    // Simple spatial median split along longest axis
    glm::vec3 minBounds, maxBounds;
    computeBounds(first, count, minBounds, maxBounds);
    
    glm::vec3 extent = maxBounds - minBounds;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;
    
    float splitPos = (minBounds[axis] + maxBounds[axis]) * 0.5f;
    
    // Partition primitives around split position
    int left = first;
    int right = first + count - 1;
    
    while (left <= right) {
        // Find centroid of left primitive
        const Primitive& leftPrim = primitives[left];
        
        glm::vec3 leftMin, leftMax;
        if (leftPrim.type == PRIM_GEOM) {
            leftMin = computeGeomBounds(leftPrim.index, true);
            leftMax = computeGeomBounds(leftPrim.index, false);
        } else if (leftPrim.type == PRIM_TRIANGLE) {
            leftMin = computeTriangleBounds(leftPrim.index, true);
            leftMax = computeTriangleBounds(leftPrim.index, false);
        }
        
        float leftCentroid = (leftMin[axis] + leftMax[axis]) * 0.5f;
        
        if (leftCentroid < splitPos) {
            left++;
        } else {
            // Swap with right primitive
            std::swap(primitives[left], primitives[right]);
            right--;
        }
    }
    
    // Ensure we don't create empty partitions
    int mid = left;
    if (mid == first) mid = first + 1;
    if (mid == first + count) mid = first + count - 1;
    
    return mid;
}
#endif // USE_BVH
