
struct ray
{
    vec3 Origin;
    vec3 Direction;
    vec3 InverseDirection;
};

struct triangle
{
    vec4 PositionUvX0;
    vec4 PositionUvX1;
    vec4 PositionUvX2;
    
    vec4 NormalUvY0; 
    vec4 NormalUvY1; 
    vec4 NormalUvY2;
    
    vec4 Tangent0;
    vec4 Tangent1;  
    vec4 Tangent2;
    
    vec3 Centroid;
    float padding3; 
};


struct bvhNode
{
    vec3 AABBMin;
    float LeftChildOrFirst;
    vec3 AABBMax;
    float TriangleCount; 
};

struct indexData
{
    uint triangleDataStartInx;
    uint IndicesDataStartInx;
    uint BVHNodeDataStartInx;
    uint TriangleCount;
};

struct aabb
{
    vec3 Min;
    float pad0;
    vec3 Max;
    float pad1;
};

struct instance
{
    mat4 Transform;
    mat4 InverseTransform;
    mat4 NormalTransform;
    aabb Bounds;

    uint Shape;
    uint Index;
    uint Material;
    uint Selected;  
};

struct tlasNode
{
    vec3 AABBMin;
    uint LeftRight;
    vec3 AABBMax;
    uint BLAS;
};

struct camera
{
    mat4 Frame;
    
    float Lens;
    float Film;
    float Aspect;
    float Focus;
    
    vec3 Padding0;
    float Aperture;
    
    int Orthographic;
    ivec3 Padding;
};

struct tracingParameters
{
    int CurrentSample;
    int TotalSamples;
    int Batch;
    int Bounces;

    vec2 Pad;    
    float CurrentCamera;    
    float Clamp;    
};

struct material
{
    vec3 Emission;
    float Roughness;
    
    vec3 Colour;
    float Metallic;
    
    float Padding;
    float Anisotropy;
    float MaterialType;
    float Opacity;

    vec3 ScatteringColour;
    float TransmissionDepth;

    int EmissionTexture;
    int ColourTexture;
    int RoughnessTexture;
    int NormalTexture;
};

struct materialPoint
{
    vec3 Emission;
    vec3 Colour;
    int MaterialType;
    float Roughness, Metallic, Opacity;
    
    vec3 ScatteringColour;
    float TransmissionDepth;
    vec3 Density;
    float Anisotropy;
};

struct light
{
    int Instance;
    int CDFCount;
    int Environment;
    int CDFStart;
};

struct environment
{
    mat4 Transform;
    
    vec3 Emission;
    float pad0;

    ivec3 pad1;
    int EmissionTexture;
};
