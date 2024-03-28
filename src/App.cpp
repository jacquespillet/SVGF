#include "App.h"
#include "Window.h"

namespace gpupt
{
std::shared_ptr<application> application::Singleton = {};

application *application::Get()
{
    if(Singleton==nullptr){
        Singleton = std::make_shared<application>();
    }

    return Singleton.get();
}
    
void application::Init()
{
    Window = std::make_shared<window>(800, 600);
}

void application::Run()
{
    while(!Window->ShouldClose())
    {
        Window->PollEvents();
    }
}

void application::Cleanup()
{
    
}

}