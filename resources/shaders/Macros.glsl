#define MATERIAL_TYPE_MATTE 0
#define MATERIAL_TYPE_PBR   1
#define MATERIAL_TYPE_VOLUMETRIC   2
#define MATERIAL_TYPE_GLASS   3
#define MATERIAL_TYPE_SUBSURFACE   4


#define SAMPLING_MODE_BSDF 0
#define SAMPLING_MODE_LIGHT 1
#define SAMPLING_MODE_BOTH 2
#define SAMPLING_MODE_MIS 3

#define PI_F 3.141592653589
#define INVALID_ID -1
#define MIN_ROUGHNESS (0.03f * 0.03f)


#define MAX_LENGTH 1e30f

#define INIT()

#define MAIN()  void main()

#define GLOBAL_ID() \
    gl_GlobalInvocationID.xy

#define IMAGE_SIZE(Img) \
    imageSize(Img)


#define FN_DECL

#define INOUT(Type) inout Type

#define GET_ATTR(Obj, Attr) \
    Obj.Attr

#define textureSample texture
#define textureSampleEnv texture
#define atan2 atan