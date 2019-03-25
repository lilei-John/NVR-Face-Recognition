   /*
* Copyright(C) 2010,Hikvision Digital Technology Co., Ltd 
* 
* File   name£ºGetStream.cpp
* Discription£º
* Version    £º1.0
* Author     £ºpanyd
* Create Date£º2010_3_25
* Modification History£º
*/

#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include   <unistd.h> 
#endif
#define USE_SSE

#include <iostream>
#include "HCNetSDK.h"
#include "public.h"
#include <stdio.h>
#include <time.h>
#include "LinuxPlayM4.h"
#include <opencv2/opencv.hpp>
#include "util.h"
#include "face_detection.h"
#include "face_alignment.h"
#include "squeue.h"
#include <pthread.h>

using namespace std;
using namespace gdface;

void init_detec(string model_path);

void *detect(void* vargp);
void *show(void *vargp);

seeta::FaceDetection *pdetector;
seeta::FaceAlignment *palign;
mt::threadsafe_queue <cv::Mat> *img_queue;
mt::threadsafe_queue <cv::Mat> *img_show_queue;

LONG nPort = -1;
HWND hWnd = NULL;
DWORD dRet;
long frame_cnt=0;
cv::Mat prefrm;
cv::Rect roi;

unsigned int margin = 50;
long id_count=0;
unsigned int max_frames=0;
int consistant_thresh=1;
int min_face_size=100;

long cnt=0;
int is_out=0;
int face_cnt=0;
cv::Point last_anchor(-1,-1);

char *savedir="./data/face_cap/";
char *conf_path = "./video-face-capture/conf/config.conf";

const unsigned int nrof_th=2;
pthread_t tid[nrof_th]; 
bool is_show=true;
int pts_num = 5;

void init(char* model_path)
{
    int params[10]={0};
    int nrof_params = loadconfig(conf_path, params);
    if(nrof_params<=0)
    {
        cerr<<"error: load params"<<endl;
        return;
    }
    int last_id = get_last_id(savedir);
    cerr<<"last_id is: "<<last_id<<endl;
    id_count=last_id+1;
    max_frames=params[5];
    margin=params[6];
    consistant_thresh=params[7];
    min_face_size=params[8];

    init_detec(model_path);
    img_queue = new mt::threadsafe_queue <cv::Mat>();
    img_show_queue = new mt::threadsafe_queue <cv::Mat>();
    if(img_queue==NULL||img_show_queue==NULL)
    {
        cerr<<"error in create img_queue or img_show_queue"<<endl;
        exit(-1);
    }

    //2048x1536
    roi.y=params[2];
    roi.height = params[1]-roi.y;
    roi.x=params[3];
    roi.width=params[4]-roi.x;

    //thread init
    for(int i=0; i<nrof_th-1; i++)
    {
        pthread_create(&tid[i], NULL, detect, NULL);
    }
    if (is_show)
    {
        cv::namedWindow("show");
        cv::namedWindow("face_det");
        pthread_create(&tid[nrof_th-1], NULL, show, NULL);
    }
}
void init_detec(string model_path)
{
      pdetector = new seeta::FaceDetection((model_path+"seeta_fd_frontal_v1.0.bin").c_str());

      if(pdetector==NULL)
      {
            cerr<<"error in new FaceDetection"<<endl;
            exit(-1);
      }
      pdetector->SetMinFaceSize(60);
      pdetector->SetScoreThresh(3.8f);
      pdetector->SetImagePyramidScaleFactor(0.8f);
      pdetector->SetWindowStep(6, 6);

      palign = new seeta::FaceAlignment((model_path+"seeta_fa_v1.1.bin").c_str());
      if(pdetector==NULL)
      {
            cerr<<"error in new FaceAlignment"<<endl;
            exit(-1);
      }
}

void *show(void* vargp)
{
    while(1)
    {
        if(is_show)
        {
            cv::imshow("show", img_show_queue->wait_and_pop());
            //cout<<"show"<<endl;
            cv::waitKey(10);
        }
    }
}

void *detect(void* vargp)
{
    long t0,t1;
    double secs=0;
    double score=0;
    string savepath=savedir;
    string filename;
    cv::Rect face_rect;
    cv::Rect face_roi;

    seeta::ImageData img_data;
    seeta::FacialLandmark points[5];
    cerr<<min_face_size<<endl;

    while(1)
    {
        cv::Mat img = img_queue -> wait_and_pop();//if there is no image in queue thread chocked
        cv::Mat img_gray;
        cv::Mat img_show;
        cv::Mat img_eval;

        img_show=img.clone();
        t0 = cv::getTickCount();
        if (img.channels() != 1)
            cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
        else
            img_gray = img;

        img_data.data = img_gray.data;
        img_data.width = img_gray.cols;
        img_data.height = img_gray.rows;
        img_data.num_channels = 1;

        std::vector<seeta::FaceInfo> faces = pdetector->Detect(img_data);
        face_cnt=(int32_t)faces.size();
        cerr<<"detect face: "<<face_cnt<<endl;
        cerr<<"has saved frame: "<<cnt<<endl;

        //only process faces[0]
        if(face_cnt>1)
            face_cnt=1;

        for (int32_t i = 0; i < face_cnt; i++) {
           
            face_rect.x = max<int>(faces[i].bbox.x - margin/2, 0);
            face_rect.y = max<int>(faces[i].bbox.y - margin/2, 0);

            // face_roi.x=faces[i].bbox.x;
            // face_roi.y=faces[i].bbox.y;
            // face_roi.width=faces[i].bbox.width;
            // face_roi.height=faces[i].bbox.height;

            if(abs((faces[i].bbox.width-faces[i].bbox.height)>10)
                ||(faces[i].bbox.width<min_face_size)||(faces[i].bbox.height<min_face_size))
            {
                face_cnt=0;
                cerr<<"drop one min face"<<endl;
                continue;
            }
            face_rect.x = max<int>(faces[i].bbox.x - margin/2, 0);
            face_rect.y = max<int>(faces[i].bbox.y - margin/2, 0);

            face_rect.width = min<int>((img_data.width - faces[i].bbox.x), (faces[i].bbox.width + margin));
            face_rect.height = min<int>((img_data.height - faces[i].bbox.y), (faces[i].bbox.height + margin));

            palign->PointDetectLandmarks(img_data, faces[i], points);
            
            for (int i = 0; i<pts_num; i++)
            {
                cv::circle(img_show, cvPoint(points[i].x, points[i].y), 2, CV_RGB(0, 255, 0), CV_FILLED);
            }
            cv::rectangle(img_show, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);
        }

        if(face_cnt==1)
        {
            if(abs(face_rect.width-face_rect.height)<10)
            {
                /* code */
                if((last_anchor.x!=-1) && (last_anchor.y!=-1))
                {
                    if (abs(face_rect.x-last_anchor.x)<consistant_thresh&&abs(face_rect.y-last_anchor.y)<consistant_thresh)
                    {
                        if (is_invalid_face(points, pts_num))
                        {
                            if(access((savepath+toString(id_count)).c_str(),0)==-1)
                            {
                                mkdir((savepath+toString(id_count)).c_str(),S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH);
                            }
                            // cerr<<"debug-4"<<endl;
                            // cerr<<"w x h: "<<img.cols<<" x "<<img.rows<<endl;
                            // cerr<<"debug-5"<<endl;
                            // img_eval=img(face_roi);
                            // cerr<<"debug-6"<<endl;
                            // score=evaluation(img_eval);
                            // cerr<<"debug-7"<<endl;
                            // cout << cnt << ": " << score << endl;
                            filename = savepath + toString(id_count) + "/" + toString(cnt++) + ".png";
                            cv::imwrite(filename, img(face_rect));
                            //cerr<<"debug-1"<<endl;
                        }
                        else
                        {
                            cerr << "Drop one Invalid face" << endl;
                        }
                        
                        last_anchor.x=face_rect.x;
                        last_anchor.y=face_rect.y;
                    }
                }
                else
                {
                    last_anchor.x=face_rect.x;
                    last_anchor.y=face_rect.y;
                }
                
                if(cnt>max_frames)
                {
                    cout<<id_count<<endl;
                    id_count++;
                    last_anchor.x=-1;
                    last_anchor.y=-1;
                    img_queue->clear();
                    if(id_count > 1000)
                        exit(0);
                    cnt=0;
                    sleep(1);
                }
            }
            //cerr<<"debug-2"<<endl;
            if(is_show)
                cv::imshow("face_det", img_show);
            //cerr<<"debug-3"<<endl;
            if(is_show)
                cv::waitKey(5);
        }

        t1 = cv::getTickCount();
        secs = (t1 - t0)/cv::getTickFrequency();
        cerr << "Detections takes " << secs << " seconds " << endl;

        //release
        img_eval.release();
        img.release();
        img_gray.release();
        img_show.release();
    }
}

void CALLBACK DecCBFun(int nPort, char * pBuf, int nSize, FRAME_INFO * pFrameInfo, void * nReserved1, int nReserved2)
{
    long lFrameType = pFrameInfo->nType;

    //cout<<"in DecCBFun"<<endl;
    static string savedir="../../data/scratch/";
    //cout<<pFrameInfo->nWidth<<"x"<<pFrameInfo->nHeight<<endl;

    if (lFrameType == T_YV12)
    {
        //use opencv decode
        cv::Mat dst(pFrameInfo->nHeight,pFrameInfo->nWidth,CV_8UC3);
        cv::Mat src(pFrameInfo->nHeight + pFrameInfo->nHeight/2,pFrameInfo->nWidth,CV_8UC1,pBuf);
        cv::cvtColor(src,dst,CV_YUV2BGR_YV12);
        dst=dst(roi);
        if(is_show)
        {
             img_show_queue->push(dst.clone());
        }

        cv::Mat dst_gray;
        cv::cvtColor(dst, dst_gray, CV_BGR2GRAY);
        //cout<<"image size wxh"<<dst.cols<<"x"<<dst.rows<<endl;
        if(frame_cnt > 0)
        {
            //img_queue->push(dst.clone());
            if(motionDetec(dst_gray, prefrm))
            {
                //cout<<"motion frame detected"<<endl;
                img_queue->push(dst.clone());
            }

            if(cnt>0 && (face_cnt==0))
            {
                if((++is_out)>60)
                {
                    is_out=0;
                    cout<<id_count<<endl;
                    last_anchor.x=-1;
                    last_anchor.y=-1;
                    img_queue->clear();
                    id_count++;
                    cnt=0;
                }
            }
            else{
                is_out=0;
               //detect some one
            }
        }

        prefrm=dst_gray.clone();
        frame_cnt++;

        // if(frame_cnt<=10)
        // {
        //      string filename = savedir+toString(frame_cnt)+".jpg";
        //      cout<<filename<<endl;
        //      cv::imwrite(filename, dst);
        // }
        // cout<<"has save "<<frame_cnt<<" frames"<<endl;

        src.release();
        dst_gray.release();
        dst.release();

    }
    else
    {
        cerr<<"Invalid Frame Type; please check the frame Type"<<endl;
        exit(1);
    }
}

void CALLBACK g_HikDataCallBack(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer,DWORD dwBufSize,DWORD dwUser)
{
    //printf("pyd---(private)Get data,the size is %d.\n", dwBufSize);
    switch (dwDataType)
    {
        case NET_DVR_SYSHEAD:    //系统头
            if (!PlayM4_GetPort(&nPort)) //获取播放库未使用的通道号
            {
                //dRet = PlayM4_GetLastError();
                printf("%s %d\n","ERROR: PlayM4_GetPort() error code: ",dRet);
                break;
            }
            cerr<< "port is "<< nPort << endl;

            if (dwBufSize > 0)
            {
                if (!PlayM4_OpenStream(nPort, pBuffer, dwBufSize, 1024 * 1024))
                {
                    dRet = PlayM4_GetLastError(nPort);
                    cerr<<"error in PlayM4_OpenStream()"<<endl;
                    break;
                }
                //设置解码回调函数 只解码不显示
                if (!PlayM4_SetDecCallBack(nPort, DecCBFun))
                {
                    dRet = PlayM4_GetLastError(nPort);
                    cerr<<"error in PlayM4_SetDecCallBack()"<<endl;
                    break;
                }
                //设置解码回调函数 解码且显示
                //if (!PlayM4_SetDecCallBackEx(nPort,DecCBFun,NULL,NULL))
                //{
                //  dRet=PlayM4_GetLastError(nPort);
                //  break;
                //}
                //打开视频解码
                if (!PlayM4_Play(nPort, hWnd))
                {
                    dRet = PlayM4_GetLastError(nPort);
                    cerr<<"error in PlayM4_Play()"<<endl;
                    break;
                }
                //打开音频解码, 需要码流是复合流
                //          if (!PlayM4_PlaySound(nPort))
                //          {
                //              dRet=PlayM4_GetLastError(nPort);
                //              break;
                //          }     
            }
            break;


    case NET_DVR_STREAMDATA:   //码流数据
        if (dwBufSize > 0 && nPort != -1)
        {
            BOOL inData = PlayM4_InputData(nPort, pBuffer, dwBufSize);
            while (!inData)
            {
                sleep(10);
                inData = PlayM4_InputData(nPort, pBuffer, dwBufSize);
                cerr << (L"PlayM4_InputData failed \n") << endl;
            }
        }
        break;
    }
}

void CALLBACK g_StdDataCallBack(int lRealHandle, unsigned int dwDataType, unsigned char *pBuffer, unsigned int dwBufSize, unsigned int dwUser)
{
    printf("pyd---(rtsp)Get data,the size is %d.\n", dwBufSize);
}



/*******************************************************************
      Function:   Demo_GetStream
   Description:   preview(no "_V30")
     Parameter:   (IN)   none  
        Return:   0--successful£¬-1--fail¡£   
**********************************************************************/
int Demo_GetStream()
{

    NET_DVR_Init();
    long lUserID;
    //login
    NET_DVR_DEVICEINFO struDeviceInfo;
    lUserID = NET_DVR_Login("10.193.5.78", 8000, "admin", "324865SEU", &struDeviceInfo);
    if (lUserID < 0)
    {
        printf("pyd1---Login error, %d\n", NET_DVR_GetLastError());
        return HPR_ERROR;
    }
    //cv::namedWindow("show");
    //Set callback function of getting stream.
    long lRealPlayHandle;
    NET_DVR_CLIENTINFO ClientInfo = {0};
#if (defined(_WIN32) || defined(_WIN_WCE))
    ClientInfo.hPlayWnd     = NULL;
#elif defined(__linux__)
    ClientInfo.hPlayWnd     = 0;
#endif

    ClientInfo.lChannel     = 1;  //channel NO
    //ClientInfo.lLinkMode  = 0x40000000; //Record when breaking network.
    ClientInfo.lLinkMode    = 0;
    ClientInfo.sMultiCastIP = NULL;

    lRealPlayHandle = NET_DVR_RealPlay(lUserID, &ClientInfo);
    if (lRealPlayHandle < 0)
    {
        printf("pyd1---NET_DVR_RealPlay_V30 error\n");
        NET_DVR_Logout(lUserID);
        NET_DVR_Cleanup();
        return HPR_ERROR;
    }
    
    //Set callback function of getting stream.
    int iRet;
    iRet = NET_DVR_SetRealDataCallBack(lRealPlayHandle, g_HikDataCallBack, 0);
    if (!iRet)
    {
        printf("pyd1---NET_DVR_RealPlay_V30 error\n");
        NET_DVR_StopRealPlay(lRealPlayHandle);
        NET_DVR_Logout(lUserID);
        NET_DVR_Cleanup();  
        return HPR_ERROR;
    }


// #ifdef _WIN32
//     Sleep(5000);  //millisecond
// #elif  defined(__linux__)
//     sleep(500);   //second
// #endif

    for(int i=0; i<nrof_th; i++)
    {
        pthread_join(tid[i],NULL);
    }
    
    while(1)
    {
        sleep(500);
    }
    //stop
    NET_DVR_StopRealPlay(lRealPlayHandle);
    NET_DVR_Logout(lUserID);
    NET_DVR_Cleanup();
    return HPR_OK;

}

void CALLBACK g_ExceptionCallBack(DWORD dwType, LONG lUserID, LONG lHandle, void *pUser)
{
    char tempbuf[256] = {0};
    switch(dwType) 
    {
    case EXCEPTION_RECONNECT:			
        printf("pyd----------reconnect--------%d\n", time(NULL));
        break;
    default:
        break;
    }
};


