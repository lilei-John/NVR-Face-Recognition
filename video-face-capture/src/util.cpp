#include "util.h"
#include <stdio.h>  
#include <unistd.h>  
#include <dirent.h>  
#include <stdlib.h>  

extern long frame_cnt;
const int thresh1=8;
const int thresh2=8;
const int thresh3=8;

bool is_invalid_face(seeta::FacialLandmark *point, int pts_num)
{
    int eye_cent=(point[0].x+point[1].x)/2;
    int eye_diff=abs(point[0].y-point[1].y);
    int mouse_cent=(point[3].x + point[4].x)/2;
    int mouse_diff=abs(point[3].y - point[4].y);
    int mouse_line=(point[3].y + point[4].y)/2;
    int dist_mouse_nose=abs(point[2].y-mouse_line);

    if(eye_diff>thresh3||mouse_diff>thresh3)
        return false;
    if(abs(eye_cent-point[2].x)>thresh1)
        return false;
    if(abs(mouse_cent-point[2].x)>thresh2)
        return false;
    if(dist_mouse_nose<thresh1)
        return false;
    return true;
}

double evaluation(Mat &img)
{
    int row,col;
    long ret=0;
    Mat img_blur;
    Mat img_gray;
    row=img.rows;
    col=img.cols;
    cout<<row<<" x "<<col<<endl;
    GaussianBlur(img, img_blur, Size(3,3), 0, 0, BORDER_DEFAULT );
    cvtColor(img_blur, img_gray, CV_RGB2GRAY );
    for(int i=0;i<row-1;i++)
    {
        for(int j=0; j<col-1; j++)
        {
            ret+=abs(img_gray.at<uchar>(i+1,j)-img_gray.at<uchar>(i,j));
            ret+=abs(img_gray.at<uchar>(i,j+1)-img_gray.at<uchar>(i,j));
        }
    }
    return double(ret)/(col*row);
}
string toString(long a)  
{  
    stringstream ss;  
    ss<<a;  
    string result;  
    ss>>result;  
    return result;  
}

vector<string> split(const string& str, const string& delim) {  
    vector<string> res;  
    if("" == str) return res;  
    //先将要切割的字符串从string类型转换为char*类型  
    char * strs = new char[str.length() + 1] ; //不要忘了  
    strcpy(strs, str.c_str());   
  
    char * d = new char[delim.length() + 1];  
    strcpy(d, delim.c_str());  
  
    char *p = strtok(strs, d);  
    while(p) {  
        string s = p; //分割得到的字符串转换为string类型  
        res.push_back(s); //存入结果数组  
        p = strtok(NULL, d);  
    }  
  
    return res;  
} 

char buf[20]={0};
int loadconfig(char* conf, int* params)
{
    ifstream ifs(conf);
    if(!ifs.is_open())
    {
        cerr<<"open config file failed: "<<conf<<endl;
        return -1;
    }
    int ret=0;
    while(ifs>>buf)
    {
        *(params++) = atoi(split(string(buf),"=")[1].c_str());
        ++ret;
    }
    return ret;
}
/* Show all files under dir_name , do not show directories ! */  
void showAllFiles( const char * dir_name )  
{  
    // check the parameter !  
    if( NULL == dir_name )  
    {  
        cerr<<" dir_name is null ! "<<endl;  
        return;  
    }  
  
    // check if dir_name is a valid dir  
    struct stat s;  
    lstat( dir_name , &s );  
    if( ! S_ISDIR( s.st_mode ) )  
    {  
        cerr<<"dir_name is not a valid directory !"<<endl;  
        return;  
    }  
      
    struct dirent * filename;    // return value for readdir()  
    DIR * dir;                   // return value for opendir()  
    dir = opendir( dir_name );  
    if( NULL == dir )  
    {  
        cerr<<"Can not open dir "<<dir_name<<endl;  
        return;  
    }  
    cerr<<"Successfully opened the dir !"<<endl;  
      
    /* read all the files in the dir ~ */  

    while( ( filename = readdir(dir) ) != NULL )  
    {  
        // get rid of "." and ".."  
        if( strcmp( filename->d_name , "." ) == 0 ||   
            strcmp( filename->d_name , "..") == 0    )  
            continue;  
        cout<<atoi(filename ->d_name)<<endl;  
    }  

}   

int get_last_id( const char * dir_name )  
{  
    // check the parameter !  
    if( NULL == dir_name )  
    {  
        cerr<<" dir_name is null ! "<<endl;  
        return -1; 
    }  
  
    // check if dir_name is a valid dir  
    struct stat s;  
    lstat( dir_name , &s );  
    if( ! S_ISDIR( s.st_mode ) )  
    {  
        cerr<<"dir_name is not a valid directory !"<<endl;  
        return -1;
    }  
      
    struct dirent * filename;    // return value for readdir()  
    DIR * dir;                   // return value for opendir()  
    dir = opendir( dir_name );  
    if( NULL == dir )  
    {  
        cerr<<"Can not open dir "<<dir_name<<endl;  
        return -1;
    }  
    cerr<<"Successfully opened the dir !"<<endl;  
      
    /* read all the files in the dir ~ */  
    int max_id=-1;
    int file_id;
    while( ( filename = readdir(dir) ) != NULL )  
    {  
        // get rid of "." and ".."  
        if( strcmp( filename->d_name , "." ) == 0 ||   
            strcmp( filename->d_name , "..") == 0    )  
            continue;  
        file_id=atoi(filename ->d_name);
        if(file_id>max_id)
            max_id=file_id;
        //cout<<atoi(filename ->d_name)<<endl;  
    }  
    return max_id;
    
} 
bool motionDetec(Mat &now_frm_gray, Mat &pre_frm_gray,int thresh, int thresh_cnt)
{
    Mat diff;  
    const int ch=diff.channels();
    int width, height;
    int cnt=0;
    uchar* p;
    //static string savedir="/home/xuhao/temp/data/";
    //imshow("diff", diff);  
    //Mat diff_thresh;  
    //threshold(diff, diff_thresh, thresh, 1, CV_THRESH_BINARY);  
    //imshow("diff_thresh", diff_thresh);

    absdiff(now_frm_gray, pre_frm_gray, diff);  
    //string filename = savedir+toString(frame_cnt)+".jpg";

    width=diff.cols*ch;
    height=diff.rows;

    //imwrite(filename, diff);

    for(int i=0; i<height; i++)
    {
        p=diff.ptr<uchar>(i);

        for(int j=0; j<width; j++)
        {
            if(p[j]>=thresh) cnt++;
            //cout<<"debug"<<endl;
        }
    }

    //cout<<"cnt is: "<<cnt<<endl;
    if(cnt > thresh_cnt)
        return true;
    else
        return false;
}