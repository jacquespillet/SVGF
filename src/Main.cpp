#include <stdio.h>
#include "App.h"

using namespace gpupt;

int main()
{
    application *App = application::Get();
    App->Init();
    App->Run();
    App->Cleanup();    
}