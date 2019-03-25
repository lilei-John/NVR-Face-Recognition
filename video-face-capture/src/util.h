#ifndef _UTIL_H_
#define _UTIL_H_

#include <string.h>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <sys/stat.h> 　
#include <sys/types.h> 　
#include <unistd.h>
//#include <direct.h>
#include <X11/Xlib.h>
#include "face_alignment.h"

using namespace cv;
using namespace std;

string toString(long a);
bool motionDetec(Mat &now_frm, Mat &pre_frm, int thresh=70, int thresh_cnt=800);
int loadconfig(char * conf, int* params);
vector<string> split(const string& str, const string& delim);
void showAllFiles( const char * dir_name );
int get_last_id(const char* dir_name);
double evaluation(Mat& img);
bool is_invalid_face(seeta::FacialLandmark *point, int pts_num);

#endif