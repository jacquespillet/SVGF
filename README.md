#Build
## Requirements : 
    Visual Studio (Tested only on Visual Studio 2019)
    Cuda installed on the system
    NVidia GPU (for Cuda)
    CUDA_PATH environment variable set (example  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8" )

## Commands : 
```
### Clone the repo and checkout to the latest branch
git clone --recursive https://github.com/jacquespillet/gpupt_blog.git
cd gpupt_blog
git checkout Part_13

### Generate the solution
mkdir build
cd build
cmake ../

### Build
cd ..
BuildDebug.bat / BuildRelease.bat

First build may take a while because it's going to build all the dependencies with the project.

```

# GPU Path Tracer

This is the repository accompanying the blog post series "Simple GPU Path Tracing". It contains all the code that we write throughout the series. Each branch in this repo corresponds to a blog post.

Here's a summary of all the episodes in the series :
[Simple GPU Path Tracing : Introduction ](https://jacquespillet.blogspot.com/2024/03/blog-post.html)

[Simple GPU Path Tracing, Part. 1 : Project Setup](https://jacquespillet.blogspot.com/2024/03/simple-gpu-path-tracing-part-1-project.html)

[Simple GPU Path Tracing, Part. 1.1 : Adding a cuda backend to the project](https://jacquespillet.blogspot.com/2024/03/simple-gpu-path-tracing-part-11-adding.html)

[Simple GPU Path Tracing, Part. 2.0 : Scene Representation](https://jacquespillet.blogspot.com/2024/03/simple-gpu-path-tracing-part-20-scene.html)

[Simple GPU Path Tracing, Part. 2.1 : Acceleration structure](https://jacquespillet.blogspot.com/2024/03/simple-gpu-path-tracing-part-21.html)

[Simple GPU Path Tracing, Part. 3.0 : Path Tracing Basics](https://jacquespillet.blogspot.com/2024/04/simple-gpu-path-tracing-part-21-path.html)

[Simple GPU Path Tracing, Part. 3.1 : Matte Material](https://jacquespillet.blogspot.com/2024/04/simple-gpu-path-tracing-part-31-matte.html)

[Simple GPU Path Tracing, Part. 3.2 : Physically Based Material](https://jacquespillet.blogspot.com/2024/04/simple-gpu-path-tracing-part-32-more.html)

[Simple GPU Path Tracing, Part. 3.4 : Small Improvements, Camera and wrap up](https://jacquespillet.blogspot.com/2024/04/simple-gpu-path-tracing-part-34-small.html)

[Simple GPU Path Tracing, Part. 4.0 : Mesh Loading](https://jacquespillet.blogspot.com/2024/04/simple-gpu-path-tracing-part-40-mesh.html)

[Simple GPU Path Tracing, Part. 4.1 : Textures](https://jacquespillet.blogspot.com/2024/04/simple-gpu-path-tracing-part-41-textures.html)

[Simple GPU Path Tracing, Part. 4.2 : Normal Mapping & GLTF Textures](https://jacquespillet.blogspot.com/2024/04/simple-gpu-path-tracing-part-42-normal.html)

[Simple GPU Path Tracing, Part. 5.0 : Sampling lights](https://jacquespillet.blogspot.com/2024/04/simple-gpu-path-tracing-part-50.html)

[Simple GPU Path Tracing, Part 6 : GUI](https://jacquespillet.blogspot.com/2024/04/simple-gpu-path-tracing-part-6-gui.html)

[Simple GPU Path Tracing, Part 7.0 : Transparency](https://jacquespillet.blogspot.com/2024/04/simple-gpu-path-tracing-part-70.html)

[Simple GPU Path Tracing, Part 7.1 : Volumetric materials](https://jacquespillet.blogspot.com/2024/04/simple-gpu-path-tracing-part-71.html)

[Simple GPU Path Tracing, Part 7.1 : Refractive material](https://jacquespillet.blogspot.com/2024/04/simple-gpu-path-tracing-part-71_9.html)

[Simple GPU Path Tracing, Part 8 : Denoising](https://jacquespillet.blogspot.com/2024/04/simple-gpu-path-tracing-part-8-denoising.html)

[Simple GPU Path Tracing, Part 9 : Environment Lighting](https://jacquespillet.blogspot.com/2024/04/simple-gpu-path-tracing-part-9.html)

[Simple GPU Path Tracing, Part 10 : Little Optimizations](https://jacquespillet.blogspot.com/2024/04/simple-gpu-path-tracing-part-10-little.html)

[Simple GPU Path Tracing, Part 11 : Multiple Importance Sampling](https://jacquespillet.blogspot.com/2024/04/simple-gpu-path-tracing-part-11.html)


---
  

[Here](https://github.com/jacquespillet/gpupt_blog/tree/Part_13/resources/Gallery) are some renders done with the path tracer resulting from the tutorials :
![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/Teapot.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/Bottle.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/AnimeClassRoom.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/BaseScene.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/Cathedral_0.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/Robot.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/Sponza.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/Vokselia_2.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/Sculpture_All.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/Bathroom.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/Cathedral.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/Breakfast_Room_2.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/Coffee.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/ConferenceRoom.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/Lost_Empire_1.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/Rhetorician.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/Rhetorician_Glass.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/Rhetorician_Volume.png?raw=true)

![Image](https://github.com/jacquespillet/gpupt_blog/blob/Part_13/resources/Gallery/Sponza_1.png?raw=true)

--- 
[Here](https://github.com/jacquespillet/gpupt_blog/releases/download/Vendor/Scenes.zip) are the scenes that were used for those renders.

---

[Here](https://raw.githubusercontent.com/jacquespillet/gpupt_blog/Part_13/resources/Gallery/Credits.txt?token=GHSAT0AAAAAACQGK7WP7XR5AOFYFSPGP44AZRKLF3A) are the credits for each scene :

```
[Anime Class Room by AnixMoonLight](https://sketchfab.com/3d-models/anime-class-room-4faa1d57304d446995bc3a01af763239)

[Salle de bain from McGuire Computer Graphics Archive, Model by Nacimus Ait Cherif](https://casual-effects.com/data)

[Bedroom from McGuire Computer Graphics Archive, Model by fhernand](https://casual-effects.com/data)

[Ship in a bottle by Loïc Norgeot](https://sketchfab.com/3d-models/ship-in-a-bottle-9ddbc5b32da94bafbfdb56e1f6be9a38)

[Breakfast Room from McGuire Computer Graphics Archive, Model by Wig42](https://casual-effects.com/data)

[Breakfast Room from McGuire Computer Graphics Archive, Model by Wig42](https://casual-effects.com/data)

[Cathedral by patrix](https://sketchfab.com/3d-models/cathedral-faed84a829114e378be255414a7826ca)

[Coffee Maker from cekuhnen](www.blendswap.com/user/cekuhnen)

[Conference Room from McGuire Computer Graphics Archive, Model by Anat Grynberg and Greg Ward](https://casual-effects.com/data)

[Fireplace Room from McGuire Computer Graphics Archive, Model by Wig42](https://casual-effects.com/data)

[List Empire from McGuire Computer Graphics Archive, Model by Morgan McGuire](https://casual-effects.com/data)

[Rhetorician Maker from engine9](https://sketchfab.com/3d-models/rhetorician-a89f035291d843069d73988cc0e25399)

[Robo Maker from Artem Shupa-Dubrova](https://sketchfab.com/3d-models/robo-obj-pose4-uaeYu2fwakD1e1bWp5Cxu3XAqrt)

[Loie Fuller from Minneapolis Institute of Art, by Joseph Kratina](https://sketchfab.com/3d-models/loie-fuller-sculpture-by-joseph-kratina-05ba00cefb5b4a7e8cecda2a91d6a568)

[Sea Keep "Lonely Watcher" by Artjoms Horosilovs](https://sketchfab.com/3d-models/sea-keep-lonely-watcher-09a15a0c14cb4accaf060a92bc70413d)

[Sea House by Alyona Shek](https://sketchfab.com/3d-models/sea-house-bc4782005e9646fb9e6e18df61bfd28d)

[Sponza from McGuire Computer Graphics Archive, by Frank Meinl, Crytek](https://casual-effects.com/data/)

[Renderman's Teapot by Dylan Sisson and Leif Pedersen](https://renderman.pixar.com/official-swatch)

[Vokselia Spawn from McGuire Computere Graphics Archive, Model by Vokselia](https://casual-effects.com/data/)


```

--- 
Helpful resources that helped creating the tutorials

[Physically Based Rendering online edition](https://pbr-book.org/)
[Jacco Bikker Blog](https://jacco.ompf2.com/), especially the posts about BVH that we're using in our code
[A Graphics Guy's Note](https://agraphicsguynotes.com/posts/)
[Scratchapixel](https://www.scratchapixel.com/)
[Crash Course in BRDF Implementation](https://boksajak.github.io/blog/BRDF)
