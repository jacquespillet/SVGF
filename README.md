
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

Hello

```

--- 
Helpful resources that helped creating the tutorials

