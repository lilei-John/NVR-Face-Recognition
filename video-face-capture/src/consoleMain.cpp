 /*
* main()
*/

#ifndef __APPLE__

#include <stdio.h>
#include <iostream>
#include "GetStream.h"
#include "util.h"
using namespace std;

int main()
{
    NET_DVR_Init();
    XInitThreads();
    //Demo_SDK_Version();

    char* sdk_log_path = "./video-face-capture/logs/sdkLog";
    char* seeta_face_model_path = "./video-face-capture/model/";

    NET_DVR_SetLogToFile(3, sdk_log_path);
    
    init(seeta_face_model_path);

    Demo_GetStream();
    
    while(1)
    {

    }
    return 0;
}

#endif
