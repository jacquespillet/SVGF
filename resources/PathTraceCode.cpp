
MAIN()
{
    INIT()
    
    ivec2 ImageSize = IMAGE_SIZE(RenderImage);
    int Width = ImageSize.x;
    int Height = ImageSize.y;

    uvec2 GlobalID = GLOBAL_ID();
    if (GlobalID.x < Width && GlobalID.y < Height) {    
        ivec2 ImageSize = IMAGE_SIZE(RenderImage);
        vec2 UV = vec2(GLOBAL_ID()) / vec2(ImageSize);
        imageStore(RenderImage, ivec2(GLOBAL_ID()), vec4(UV, 0, 1));    
    }
}