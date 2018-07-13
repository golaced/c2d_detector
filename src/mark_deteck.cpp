
#include "mark_deteck.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <string>
#include <sstream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <opencv2/core/eigen.hpp>
#define ADAPTIVE_THRESH_SIZE 35
#define APPROX_POLY_EPS 0.01
#define MARKER_CELL_SIZE 10
#define MARKER_SIZE (7*MARKER_CELL_SIZE)

using namespace std;
using namespace cv;
using namespace Eigen; 
#if USE_ARUCO
using namespace aruco;
#endif
//========================================Class Marker=====================================
Marker1::Marker1(void)
{
	m_id = -1;
	m_corners.resize(4, Point2f(0.f,0.f));
}


Marker1::Marker1(int m)
{
  cout<<"init color tag"<<endl;
  c_corners[0].x=-m/2.; c_corners[0].y=m/2.;
  c_corners[1].x= m/2.; c_corners[1].y=m/2.;
  c_corners[2].x= m/2.; c_corners[2].y=-m/2.;
  c_corners[3].x=-m/2.; c_corners[3].y=-m/2.;
  c_corners[4].x=  0; c_corners[4].y=0;
 }
 
 
Marker1::Marker1(int _id, cv::Point2f _c0, cv::Point2f _c1, cv::Point2f _c2, cv::Point2f _c3)
{
	m_id = _id;

	m_corners.reserve(4);
	m_corners.push_back(_c0);
	m_corners.push_back(_c1);
	m_corners.push_back(_c2);
	m_corners.push_back(_c3);
}

void Marker1::drawToImage(cv::Mat& image, cv::Scalar color, float thickness,Point3f pos_2d,Point3f att_2d)
{
	circle(image, m_corners[0], thickness*2, color, thickness);
	circle(image, m_corners[1], thickness, color, thickness);
	line(image, m_corners[0], m_corners[1], color, thickness, CV_AA);
	line(image, m_corners[1], m_corners[2], color, thickness, CV_AA);
	line(image, m_corners[2], m_corners[3], color, thickness, CV_AA);
	line(image, m_corners[3], m_corners[0], color, thickness, CV_AA);
	
	Point text_point =  m_corners[2];
	text_point.x = 20;
	text_point.y = 20;

	stringstream ss;
	ss <<"2D:"<<pos_2d;

	putText(image, ss.str(), text_point, FONT_HERSHEY_SIMPLEX,0.5, Scalar(0,0,250),2);
        text_point.x = 22;
	text_point.y = 240-44;
	stringstream s1s;
	s1s <<"2D_att"<<att_2d;
	putText(image, s1s.str(), text_point, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255),2);
}


void drawToImage_single(cv::Mat& image, std::vector<cv::Point2f> in_conner,cv::Scalar color, float thickness)
{
	circle(image, in_conner[0], thickness*2, color, thickness);
	circle(image, in_conner[1], thickness, color, thickness);
	line(image, in_conner[0], in_conner[1], color, thickness, CV_AA);
	line(image, in_conner[1], in_conner[2], color, thickness, CV_AA);
	line(image, in_conner[2], in_conner[3], color, thickness, CV_AA);
	line(image, in_conner[3], in_conner[0], color, thickness, CV_AA);
	
	Point text_point = in_conner[0] + in_conner[2];
	text_point.x /= 2;
	text_point.y /= 2;
}




void Marker1::estimateTransformToCamera(vector<Point3f> corners_3d, cv::Mat& camera_matrix, cv::Mat& dist_coeff, cv::Mat& rmat, cv::Mat& tvec)
{
	Mat rot_vec;
	bool res = solvePnP(corners_3d, m_corners, camera_matrix, dist_coeff, rot_vec, tvec);
	Rodrigues(rot_vec, rmat);
}



//====================================Class MarkerRecognizer================================
MarkerRecognizer::MarkerRecognizer()
{
	//±ê×ŒMarker×ø±ê£¬ÄæÊ±Õë
	m_marker_coords.push_back(Point2f(0,0));
	m_marker_coords.push_back(Point2f(0, MARKER_SIZE-1));
	m_marker_coords.push_back(Point2f(MARKER_SIZE-1, MARKER_SIZE-1));
	m_marker_coords.push_back(Point2f(MARKER_SIZE-1, 0));
}


void MarkerRecognizer:: init_marker(float c_2d,float c_color, float r1, float r2 )
{
  coner_2D[0].x=-c_2d/2.; coner_2D[0].y=c_2d/2.;
  coner_2D[1].x= -c_2d/2.; coner_2D[1].y=-c_2d/2.;
  coner_2D[2].x= c_2d/2.; coner_2D[2].y=-c_2d/2.;
  coner_2D[3].x=c_2d/2.; coner_2D[3].y=c_2d/2.;
  
  coner_color[0].x=-c_color/2.; coner_color[0].y=c_color/2.;
  coner_color[1].x= -c_color/2.; coner_color[1].y=-c_color/2.;
  coner_color[2].x= c_color/2.; coner_color[2].y=-c_color/2.;
  coner_color[3].x=c_color/2.; coner_color[3].y=c_color/2.;
  r2d=c_2d;
  k_circle=r1/r2;
  cr1=r1;cr2=r2;
}

float MarkerRecognizer::To_180_degrees(float x)
{
	return (x>180?(x-360):(x<-180?(x+360):x));
}

void MarkerRecognizer::getCameraPos(cv::Mat Rvec, cv::Mat Tvec, cv::Point3f &pos,cv::Point3f &att)
{
      Mat R(3, 3, CV_32FC1);
      cv::Rodrigues (Rvec, R ); // rÎªÐý×ªÏòÁ¿ÐÎÊœ£¬ÓÃRodrigues¹«Êœ×ª»»ÎªŸØÕó
      //cout<<"R="<<endl<<R<<endl;
      //cout<<"t="<<endl<<Tvec<<endl;
	double r11 = R.ptr<double>(0)[0];
	double r12 = R.ptr<double>(0)[1];
	double r13 = R.ptr<double>(0)[2];
	double r21 = R.ptr<double>(1)[0];
	double r22 = R.ptr<double>(1)[1];
	double r23 = R.ptr<double>(1)[2];
	double r31 = R.ptr<double>(2)[0];
	double r32 = R.ptr<double>(2)[1];
	double r33 = R.ptr<double>(2)[2];
       Matrix3d r1; 
       r1<< r11,r12,r13,r21,r22,r23,r31,r32,r33;
    //  std::cout << "Here is re:\n" << r1 << std::endl;  
      Vector3d t1;
      double t10=Tvec.at<double>(0);
      double t11=Tvec.at<double>(1);
      double t12=Tvec.at<double>(2);
      t1<<t10,t11,t12;
      //cout<< "Here is te:\n" << t1 << std::endl;  
      Vector3d P_oc;
      P_oc = -r1.inverse()*t1;
    pos.x = P_oc(0)*1;
    pos.y = P_oc(1)*1;
    pos.z = P_oc(2)*1;
    float E1,E2,E3,dlta,pi=3.1415926;
    if (r13 == 1 ||r13== -1){
    E3 = 0; 
    dlta = atan2(r12,r13);
	if( r13== -1){
	E2 = pi/2;
	E1 = E3 + dlta;}
	else{
	E2 = -pi/2;
	E1 = -E3 + dlta;}
    }
    else{
    E2 = - asin(r13);
    E1 = atan2(r23/cos(E2), r33/cos(E2));
    E3 = atan2(r12/cos(E2), r11/cos(E2));
    }
    att.x=To_180_degrees(180-E1*57.3);
    att.y=E2*57.3;
    att.z=-E3*57.3;

    /*Eigen::Matrix3d eigen_r;
    cv2eigen(R,eigen_r);
    Eigen::AngleAxisd angle(eigen_r);
    Eigen::Matrix<double,3,1> t;
    cv::cv2eigen(Tvec, t);
    t = -1 * angle.inverse().matrix() *t;
    //pos.x = t(0);
    //pos.y = -t(1);
    //pos.z = t(2);*/

}
#if USE_ARUCO
void MarkerRecognizer::getAttitudea(aruco::Marker marker, Point3f &attitude)
{
	double pos[3] = { 0 };
	double ori[4] = { 0 };

	double q0, q1, q2, q3;

	marker.OgreGetPoseParameters(pos, ori);
	pos[0] = -pos[0];
	pos[1] = -pos[1];

	q0 = ori[0]; q1 = ori[1]; q2 = ori[2]; q3 = ori[3];

	attitude.x = atan2(2 * (q0 * q1 + q2 * q3), -1 + 2 * (q1 * q1 + q2 * q2)) *57.3f;
	attitude.y = asin(2 * (q1 * q3 - q0 * q2)) *57.3f;
	attitude.z = -atan2(2 * (-q1 * q2 - q0 * q3), 1 - 2 * (q0 * q0 + q1 * q1)) *57.3f;
}

void MarkerRecognizer::getCameraPosa(cv::Mat Rvec, cv::Mat Tvec, cv::Point3f &pos)
{
     Mat Rot(3, 3, CV_32FC1);
    Rodrigues(Rvec, Rot);
    Eigen::Matrix3d eigen_r;
   cv2eigen(Rot,eigen_r);

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    Eigen::AngleAxisd angle(eigen_r);
    T = angle.inverse();
    Eigen::Matrix<double,3,1> t;
    cv::cv2eigen(Tvec, t);
    t = -1 * angle.inverse().matrix() *t;
    T(0, 3) = t(0);
    T(1, 3) = t(1);
    T(2, 3) = t(2);
    pos.x = t(0);
    pos.y = t(1);
    pos.z = t(2);
    Rot = Rot.t();  // rotation of inverse
    Mat pos_camera = -Rot * Tvec; // translation of inverse
    float k=0.65;
    //pos.x = -pos_camera.at<float>(2, 0)*k;
    //pos.y = -pos_camera.at<float>(0, 0)*k;
    //pos.z = pos_camera.at<float>(1, 0)*k;

}
#endif
void MarkerRecognizer::estimateTransformToCamera_all(cv::Mat cimage,cv::Mat& rmat, cv::Mat& tvec)
{       int i,j;
	Mat rot_vec;
        
	for(i=0;i<m_markers.size();i++){
	  vector<Point3f> pts_3d;
	  vector<Point2f> pts_2d;
	  for(j=0;j<4;j++){
	  pts_3d.push_back ( Point3f ( coner_2D[j].x,coner_2D[j].y,0 ) );
	  pts_2d.push_back ( Point2f( m_markers[i].m_corners[j].x,m_markers[i].m_corners[j].y) );
	  }
	 // cout << "World Point= " << endl << pts_3d << endl;cout << "Figure Point= " << endl << pts_2d<< endl;
	   bool res = solvePnP(pts_3d, pts_2d, camera_matrix, distortion_coefficients, rot_vec, tvec);
	   Point3f pos_c,acc_c;
	   getCameraPos(rot_vec,tvec,pos_c,acc_c);
	  // cout<<"2Dposition:"<<pos_c.x<<" "<<pos_c.y<<" "<<pos_c.z<<" 2Dcattitude:"<<acc_c.x<<" "<<acc_c.y<<" "<<acc_c.z<<endl;
           center_2d.x=(m_markers[0].m_corners[0].x+m_markers[0].m_corners[1].x+
	   m_markers[0].m_corners[2].x+m_markers[0].m_corners[3].x)/4;
	   center_2d.y=(m_markers[0].m_corners[0].y+m_markers[0].m_corners[1].y+
	   m_markers[0].m_corners[2].y+m_markers[0].m_corners[3].y)/4;
	   att_2d.x=acc_c.x;
	   att_2d.y=acc_c.y;
	   att_2d.z=acc_c.z;
	   pos_2d.x=pos_c.x;//*cos(att_2d.z/57.3)+pos_c.y*-sin(att_2d.z/57.3);
	   pos_2d.y=pos_c.y;//*sin(att_2d.z/57.3)+pos_c.y*cos(att_2d.z/57.3);
	   pos_2d.z=pos_c.z;

	// stringstream  uvss[4];	
	// Point2f uv_p[4]; Point3f  err[4];
	// ProjectCheck(Vec3f(acc_c.x,acc_c.y,acc_c.z), tvec,Point3f( coner_2D[2].x, coner_2D[2].y,0), Point(0,0) ,uv_p[3],err[3]);
	// uvss[3]<<"2";
	// putText(cimage, uvss[3].str(), uv_p[3], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,125,125),2);
	//  uvss[2]<<tvec;
	 //putText(cimage, uvss[2].str(),Point(2,22), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,125),1);
	 /// uvss[0]<<acc_c;
	 //putText(cimage, uvss[0].str(),Point(2,66), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,125),1);
	}
}

int MarkerRecognizer::update(Mat& in, int min_size, int min_side_length)
{

static int init;
#if USE_ARUCO
static CameraParameters CamParam;
static MarkerDetector MDetector;
#endif
if(!init){init=1;
#if USE_ARUCO
string cameraParamFileName("/home/odroid/catkin_ws2/src/w2c_detector/CAM_320.xml");
CamParam.readFromXMLFile(cameraParamFileName);
cout << CamParam.CameraMatrix << endl;
cout << CamParam.Distorsion << endl;
#endif
}
	CV_Assert(!in.empty());

	Mat img_rgb,img_gray,show,InImage;
	in.copyTo(img_rgb); in.copyTo(show);in.copyTo(InImage);
        cvtColor(img_rgb, img_gray, CV_BGRA2GRAY);
	vector<Marker1> possible_markers;
	static float distance_r;
	//--------------2D code
	#if !USE_ARUCO
	center_2d.x=center_2d.y=0;
	markerDetect(img_rgb, possible_markers, min_size, min_side_length);
	markerRecognize_2D(img_rgb, possible_markers, m_markers);
	markerRefine(img_gray, m_markers);	
	#else
	Markersa.clear();aruco_size=0;
	if(distance_r<0.5||1)
	MDetector.detect(img_gray, Markersa, CamParam, r2d);
	//for (unsigned int i = 0; i < Markersa.size(); i++)
        //{
        //    Markersa[i].draw(InImage, Scalar(0, 0, 255), 2);

        //}
	Point3f pos_aruco(0, 0, 0);
    	Point3f att_aruco(0, 0, 0);
	center_2d.x=center_2d.y=0;
	if(Markersa.size()){
	for(int j=0;j<Markersa.size();j++)
	 if(Markersa[j].id==1){
		getCameraPosa(Markersa[j].Rvec, Markersa[j].Tvec, pos_aruco);//camera to local marker o dis
		getAttitudea(Markersa[j], att_aruco);
		center_2d.x=Markersa[j].getCenter().x;
		center_2d.y=Markersa[j].getCenter().y;
		}
	}
	att_2d.x=att_aruco.x;
	att_2d.y=att_aruco.y;
	att_2d.z=att_aruco.z;
	pos_2d.x=pos_aruco.x;
	pos_2d.y=pos_aruco.y;
	pos_2d.z=pos_aruco.z;

	aruco_size=Markersa.size();
	Mat aruco_thr;
	aruco_thr=MDetector.getThresholdedImage();
	//==cv::imshow("thes", InImage);
	#endif
	#if !USE_ARUCO
	Mat R,T;
	if(m_markers.size()+m_markers_c.size()>0)
	  estimateTransformToCamera_all(img_rgb,R,T);
	#endif
	
	//---------------Circle Tag------------------------
	static int cnt_circle;
	if(cnt_circle++>=0){cnt_circle=0;

	markerDetect_circle(img_rgb, aruco_thr, possible_markers, min_size, min_side_length);
	}else{m_markers_circle.clear();center_circle.x=center_circle.y=0;}
	if(pos_circle.z!=0)
	distance_r=pos_circle.z;
	

	if(m_markers.size()){
	circle(show,m_markers[0].m_corners[0], 2,Scalar(255,0,255),2);
	circle(show,m_markers[0].m_corners[1], 4,Scalar(255,0,255),2);
	circle(show,m_markers[0].m_corners[2], 6,Scalar(255,0,255),2);
	circle(show,m_markers[0].m_corners[3], 8,Scalar(255,0,255),2);
	}
	if(m_markers_c.size()){
	circle(show,m_markers_c[0].m_corners[0], 2,Scalar(0,25,255),2);
	circle(show,m_markers_c[0].m_corners[1], 4,Scalar(0,20,255),2);
	circle(show,m_markers_c[0].m_corners[2], 6,Scalar(0,20,255),2);
	circle(show,m_markers_c[0].m_corners[3], 8,Scalar(0,20,255),2);
	}
	//imshow("t",show);
	return m_markers.size()+m_markers_c.size()*100;
}


void AdaptiveThereshold(Mat src,Mat &dst,int T)  
{  
 
    int x1, y1, x2, y2;  
    int count=0;  
    long long sum=0;  
    int S=src.rows>>3;  //»®·ÖÇøÓòµÄŽóÐ¡S*S  
        /*°Ù·Ö±È£¬ÓÃÀŽ×îºóÓëãÐÖµµÄ±ÈœÏ¡£Ô­ÎÄ£ºIf the value of the current pixel is t percent less than this average  
                            then it is set to black, otherwise it is set to white.*/  
    src.copyTo(dst);
    int W=dst.cols;  
    int H=dst.rows;  
    long long **Argv;  
    
    Argv=new long long*[dst.rows];  
    for(int ii=0;ii<dst.rows;ii++)  
    {  
        Argv[ii]=new long long[dst.cols];  
    }  
  
    for(int i=0;i<W;i++)  
    {  
        sum=0;  
        for(int j=0;j<H;j++)  
        {             
            sum+=dst.at<uchar>(j,i);  
            if(i==0)      
                Argv[j][i]=sum;  
            else  
                Argv[j][i]=Argv[j][i-1]+sum;  
        }  
    }  
      
    for(int i=0;i<W;i++)  
    {  
        for(int j=0;j<H;j++)  
        {  
            x1=i-S/2;  
            x2=i+S/2;  
            y1=j-S/2;  
            y2=j+S/2;  
            if(x1<0)  
                x1=0;  
            if(x2>=W)  
                x2=W-1;  
            if(y1<0)  
                y1=0;  
            if(y2>=H)  
                y2=H-1;  
            count=(x2-x1)*(y2-y1);  
            sum=Argv[y2][x2]-Argv[y1][x2]-Argv[y2][x1]+Argv[y1][x1];  
              
  
            if((long long)(dst.at<uchar>(j,i)*count)<(long long)sum*(100-T)/100)  
                dst.at<uchar>(j,i)=0;  
            else  
                dst.at<uchar>(j,i)=255;  
        }  
    }  
            for (int i = 0 ; i < dst.rows; ++i)  
       {  
         delete [] Argv[i];   
       }  
         delete [] Argv;  
}  


RNG rng(12345);  

void MarkerRecognizer::markerDetect_circle(Mat& img_rgb, Mat& aruco_thr,vector<Marker1>& possible_markers, int min_size, int min_side_length)
{  
   static int min_d=55,md=3,d_wide=10,rate1=200,init,TL=60,min_box=3,ob=3;
  if(!init && 1)
   {
     namedWindow("circle", CV_WINDOW_AUTOSIZE); //create a window called "Control"
    cvCreateTrackbar("divide", "circle", &md, 10); //Hue (0 - 179)
    cvCreateTrackbar("min_d", "circle", &min_d, 200);
    cvCreateTrackbar("d_wide", "circle", &d_wide, 100);
    cvCreateTrackbar("rate1", "circle", &rate1, 1000);
    cvCreateTrackbar("TL", "circle", &TL, 255);
    cvCreateTrackbar("min_box", "circle", &min_box, 100);
    cvCreateTrackbar("ob", "circle", &ob, 10);
    init =1;
  }
  
     m_markers_circle.clear();center_circle.x=center_circle.y=0;
     Mat image1,gray1,bimage,img_gray;
     int  center_of_tag_roi[4]={0};
     Point3f pos_c,acc_c;
     int color_found=0;
     int circle_flag=0;
     vector <RotatedRect> possible_cirle,circle_found;
 
     cvtColor(img_rgb, img_gray, CV_BGRA2GRAY);
     img_gray.copyTo(bimage);
     Mat white_img(Size(img_gray.cols, img_gray.rows), CV_8UC3, Scalar::all(0));
     Mat white_img1(Size(img_gray.cols, img_gray.rows), CV_8UC3, Scalar::all(255));
     Mat white_img2(Size(img_gray.cols, img_gray.rows), CV_8UC3, Scalar::all(0));
   
     vector<vector<Point> > contours,controursa;  
     vector<  int> contours_id;  
      //ÕâŸäÏàµ±ÓÚ¶þÖµ»¯¡£ÕâžömatlabµÄÄÇŸäºÃÏñ£º Iwt = Iw>=threshold;  
      int thresh_size = (min_size/4)*2 + 1;
   
      //adaptiveThreshold(img_gray, bimage, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, thresh_size, thresh_size/3);
      //AdaptiveThereshold(img_gray, bimage,10)  ;
      //threshold(img_gray, bimage, TL, 255, THRESH_BINARY_INV|THRESH_OTSU);
      threshold(img_gray, bimage, TL, 255, THRESH_BINARY_INV|THRESH_BINARY);
      Mat element = getStructuringElement(MORPH_RECT, Size(  ob,ob ));
      morphologyEx(bimage, bimage, MORPH_CLOSE, element);	//use open operator to eliminate small patch
  
      imshow("3",bimage);
      findContours(bimage, contours, CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);  
      //morphologyEx(aruco_thr, aruco_thr, MORPH_OPEN, element);
      //findContours(aruco_thr, contours, CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
 
      //aruco_thr.copyTo(bimage);
      //imshow("aruco_thr",aruco_thr);
    Mat cimage = Mat::zeros(bimage.size(), CV_8UC3);  
    vector<  int> id_of_cirlce;  	
    for(size_t i = 0; i < contours.size(); i++)  
    {     
        size_t count = contours[i].size();  
        if( count < 88 )  
            continue;  
        //drawContours(cimage, contours, (int)i, Scalar(0,0,255), 3, 8);  
        Mat pointsf;  
	int divide=count/md;
	int dl,dh;
	int zhen=count/divide;
	int yu=count%divide;
	 if(divide<min_d){
	     dl=1;divide=count;yu=0;
	 }
	 else{
	     dl=zhen; 
	 }
        Mat(contours[i]).convertTo(pointsf, CV_32F);  
	 #define HEAD_GAP 1
         #define SAM_SEL 0 
	#if HEAD_GAP
	#if SAM_SEL
		      for(int j=0;j<5;j++){
			    vector< Point2f> temp,temp1;
			    //sample
			    float st[5]={0,          0.8,     0.6,         0.4,           0.2};
			    float ed[5]={0.8,       0.6,      0.4,        0.2,        1};
			    if(st[j]<ed[j])
			    for(int k=st[j]*count;k<ed[j]*count;k++)
				temp.push_back( Point2f(pointsf.at<float>(k,0),pointsf.at<float>(k,1)) );
			    else
			    {
				for(int k=st[j]*count;k<count;k++)
				temp.push_back( Point2f(pointsf.at<float>(k,0),pointsf.at<float>(k,1)) );
			      for(int k=0;k<ed[j]*count;k++)
				temp.push_back( Point2f(pointsf.at<float>(k,0),pointsf.at<float>(k,1)) );
			    }	
		
			    for(int k=temp.size()*0.1;k<temp.size()*0.9;k++)
				temp1.push_back( temp[k]);
			    Mat  test;
			    img_gray.copyTo(test);
			    float dis1;
			    int draw_flag=0;
			    int MAX_SAMPLE=10;
			    vector< Point2f> v[MAX_SAMPLE],v_use;
			    int for_ward_num=2          ;
			    int conner_flag=0,cnt_mask=0;
			    
			    for(int k=1+for_ward_num;k<temp1.size()-for_ward_num;k++){
			      dis1=pow(temp1[k].x-temp1[k-1].x,2)+pow(temp1[k].y-temp1[k-1].y,2);
			      float af=fastAtan2(temp1[k+for_ward_num].y-temp1[k].y,temp1[k+for_ward_num].x-temp1[k].x);
			      float ab=fastAtan2(temp1[k].y-temp1[k-for_ward_num].y,temp1[k].x-temp1[k-for_ward_num].x);
			      //cout<<"angle_dis:"<<abs(af-ab)<<"   point_dis:"<<dis1<<"   draw_flag:"<<draw_flag<<endl;
			      if((dis1>800||(abs(af-ab)>66&&abs(af-ab)<300))&&draw_flag<MAX_SAMPLE-1&&conner_flag==0){
				conner_flag=1;
				draw_flag++;
			      }
			      if(abs(af-ab)<66&&conner_flag==1)
				conner_flag=0;
				v[draw_flag].push_back(temp1[k]);
		      
			      if(k==1)
			      circle(test,temp1[k],8,Scalar(255,0,0),2);  
			      else
			      circle(test,temp1[k],2*(draw_flag+1),Scalar(255,0,0),1*(draw_flag+1));	
			      //imshow("test",test);char key = cvWaitKey(20);    while(key!=' '&&1){key = cvWaitKey(100);    }   
			    }
			  
			    int max_size=0,sel_v=0;
			    for(int k=0;k<draw_flag+1;k++)
			    {
			      if(v[k].size()>max_size)
			      {sel_v=k;max_size=v[k].size();}
			    }
			    
			    for(int k=v[sel_v].size()*0.02;k<v[sel_v].size()*0.98;k++)
				v_use.push_back( v[sel_v][k]);
			    for(int k=1;k<v_use.size();k++){
			      circle(test,v_use[k],2,Scalar(0,0,0),2);  
			    }
			  
			    #define EN_CONER_CHECK_SLOW 0
			    #if EN_CONER_CHECK_SLOW
			    imshow("test",test);
			      char key = cvWaitKey(20);    
			      while(key!=' '&&1){
			      key = cvWaitKey(100);    
			      }   
			    #endif
		      
			    if(v_use.size()>10){ 
			    //   cout<<"sel_v"<<sel_v<<" "<<v[sel_v]<<endl;
			    RotatedRect box = fitEllipse(v_use);    
			    int  flag[6];
			    flag[0]=MAX(box.size.width, box.size.height)/MIN(box.size.width, box.size.height)>3;
			    flag[1]=box.size.width* box.size.height <img_gray.cols*img_gray.rows*(float)min_box/1000.;
			    flag[2]=box.size.width* box.size.height >img_gray.cols*img_gray.rows*0.8;
			    flag[3]=MAX(box.size.width, box.size.height) > MAX(img_gray.cols,img_gray.rows)*0.8;
			    //cout<<"ellipse_flag: "<<flag[0]<<flag[1]<<flag[2]<<flag[3]<<" "<<MAX(box.size.width, box.size.height)/MIN(box.size.width, box.size.height)<<endl;
			    if( flag[0]|| flag[1]|| flag[2]||flag[3])  
			      continue;  
			    possible_cirle.push_back(box);    
			    id_of_cirlce.push_back(i);   
			    contours_id.push_back(i);
			    //»æÖÆÍÖÔ²  <---------------ALL
			    ellipse(cimage, box, Scalar(25,0,255), 1.68, CV_AA);  
			    
			    //imshow("ra",cimage);
			    }
		      }
		  #else//sam 2    WTOA
		  //find conner 
		vector<Point2f>corners;
		Mat test_wite(Size(img_gray.cols, img_gray.rows), CV_8UC3, Scalar::all(0));
		drawContours(test_wite, contours, i, Scalar::all(255), CV_FILLED);
		cvtColor(test_wite,test_wite,CV_BGR2GRAY);
		double qualityLevel = 0.4;
		double minDistance = 6;
		int blockSize = 3;
		int max_coner=6;
		bool useHarrisDetector = false;
		goodFeaturesToTrack(test_wite,corners,max_coner,qualityLevel,minDistance,Mat(),blockSize,useHarrisDetector,0.04);
		 std::vector<cv::KeyPoint> pts;
		//定义FAST特征检测类对象，阈值为40
		//cv::FastFeatureDetector fast(5);
		//fast.detect(test_wite, pts);
		//for(int s=0;s<pts.size();s++)corners.push_back(Point2f(pts[s].pt.x,pts[s].pt.y));
		int cross_point=0;
		int dead_cross=3;
		  for(int k=0;k<corners.size();k++)
		  {
		         vector< Point2f> temp,temp1;
		         Mat test;img_gray.copyTo(test); 
		        
			//for(int s=0;s<corners.size();s++)
			//{
				//circle(test,corners[s],4,Scalar(255,0,60),2);	
			//}
		        for(int a=cross_point+5;a<count;a++)
			{
			   Point2f p1(pointsf.at<float>(a,0),pointsf.at<float>(a,1));
			   for(int b=0;b<corners.size();b++){
			      float dis1=sqrt(pow(p1.x-corners[b].x,2)+pow(p1.y-corners[b].y,2));
			      if(dis1<dead_cross)
			      {cross_point=a; 
				//cout<<p1<<" "<<corners[b]<<" "<<dis1<<" "<<a<<endl;
				goto jump2;}
			    }
			   temp.push_back( p1);	  
			}
			jump2:;
		     // for(int s=0;s<temp.size();s++)
			//      circle(test,temp[s],2,Scalar(255,0,0),1);  
			//divide on points at the conner

 
		      
		        #define EN_CONER_CHECK_SLOW1 0
			 #if EN_CONER_CHECK_SLOW1
		             imshow("test",test);
			      char key = cvWaitKey(20);    
			      while(key!=' '&&1){
			      key = cvWaitKey(100);    
			      }   
		      #endif
		       if(temp.size()>10){ 
			    //   cout<<"sel_v"<<sel_v<<" "<<v[sel_v]<<endl;
			    RotatedRect box = fitEllipse(temp);    
			    int  flag[6];
			    flag[0]=MAX(box.size.width, box.size.height)/MIN(box.size.width, box.size.height)>3;
			    flag[1]=box.size.width* box.size.height <img_gray.cols*img_gray.rows*(float)min_box/1000.;
			    flag[2]=box.size.width* box.size.height >img_gray.cols*img_gray.rows*0.8;
			    flag[3]=MAX(box.size.width, box.size.height) > MAX(img_gray.cols,img_gray.rows)*0.8;
			    //cout<<"ellipse_flag: "<<flag[0]<<flag[1]<<flag[2]<<flag[3]<<" "<<MAX(box.size.width, box.size.height)/MIN(box.size.width, box.size.height)<<endl;
			    if(!( flag[0]|| flag[1]|| flag[2]||flag[3]) ){ 
			    possible_cirle.push_back(box);    
			    id_of_cirlce.push_back(i);   
			    contours_id.push_back(i);
			    //ellipse(cimage, box, Scalar(25,0,255), 1.68, CV_AA);
			     } 
			  }
		  }		
		  #define EN_STEP2 0
		  //step 2  using divie sample
		  #if EN_STEP2
		   for(int j=0;j<5;j++){
			    vector< Point2f> temp,temp1;
			    //sample
			    float st[5]={0,          0.8,     0.6,         0.4,           0.2};
			    float ed[5]={0.8,       0.6,      0.4,        0.2,        1};
			    if(st[j]<ed[j])
			    for(int k=st[j]*count;k<ed[j]*count;k++)
				temp.push_back( Point2f(pointsf.at<float>(k,0),pointsf.at<float>(k,1)) );
			    else
			    {
				for(int k=st[j]*count;k<count;k++)
				temp.push_back( Point2f(pointsf.at<float>(k,0),pointsf.at<float>(k,1)) );
			      for(int k=0;k<ed[j]*count;k++)
				temp.push_back( Point2f(pointsf.at<float>(k,0),pointsf.at<float>(k,1)) );
			    }	
		
			    for(int k=temp.size()*0.1;k<temp.size()*0.9;k++)
				temp1.push_back( temp[k]);
			    Mat  test;
			    img_gray.copyTo(test);
			    float dis1;
			    int draw_flag=0;
			    int MAX_SAMPLE=10;
			    vector< Point2f> v[MAX_SAMPLE],v_use;
			    int for_ward_num=5          ;
			    int conner_flag=0,cnt_mask=0;
			    
			    for(int k=1+for_ward_num;k<temp1.size()-for_ward_num;k++){
			      dis1=pow(temp1[k].x-temp1[k-1].x,2)+pow(temp1[k].y-temp1[k-1].y,2);
			      float af=fastAtan2(temp1[k+for_ward_num].y-temp1[k].y,temp1[k+for_ward_num].x-temp1[k].x);
			      float ab=fastAtan2(temp1[k].y-temp1[k-for_ward_num].y,temp1[k].x-temp1[k-for_ward_num].x);
			      //cout<<"angle_dis:"<<abs(af-ab)<<"   point_dis:"<<dis1<<"   draw_flag:"<<draw_flag<<endl;
			      if((dis1>800||(abs(af-ab)>66&&abs(af-ab)<300))&&draw_flag<MAX_SAMPLE-1&&conner_flag==0){
				conner_flag=1;
				draw_flag++;
			      }
			      if(abs(af-ab)<66&&conner_flag==1)
				conner_flag=0;
				v[draw_flag].push_back(temp1[k]);
		      
			      if(k==1)
			      circle(test,temp1[k],8,Scalar(255,0,0),2);  
			      else
			      circle(test,temp1[k],2*(draw_flag+1),Scalar(255,0,0),1*(draw_flag+1));	
			      //imshow("test",test);char key = cvWaitKey(20);    while(key!=' '&&1){key = cvWaitKey(100);    }   
			    }
			  
			    int max_size=0,sel_v=0;
			    for(int k=0;k<draw_flag+1;k++)
			    {
			      if(v[k].size()>max_size)
			      {sel_v=k;max_size=v[k].size();}
			    }
			    
			    for(int k=v[sel_v].size()*0.02;k<v[sel_v].size()*0.98;k++)
				v_use.push_back( v[sel_v][k]);
			    for(int k=1;k<v_use.size();k++){
			      circle(test,v_use[k],2,Scalar(0,0,0),2);  
			    }
			  
			    #define EN_CONER_CHECK_SLOW 0
			    #if EN_CONER_CHECK_SLOW
			    imshow("test",test);
			      char key = cvWaitKey(20);    
			      while(key!=' '&&1){
			      key = cvWaitKey(100);    
			      }   
			    #endif
		      
			    if(v_use.size()>10){ 
			    //   cout<<"sel_v"<<sel_v<<" "<<v[sel_v]<<endl;
			    RotatedRect box = fitEllipse(v_use);    
			    int  flag[6];
			    flag[0]=MAX(box.size.width, box.size.height)/MIN(box.size.width, box.size.height)>3;
			    flag[1]=box.size.width* box.size.height <img_gray.cols*img_gray.rows*(float)min_box/1000.;
			    flag[2]=box.size.width* box.size.height >img_gray.cols*img_gray.rows*0.8;
			    flag[3]=MAX(box.size.width, box.size.height) > MAX(img_gray.cols,img_gray.rows)*0.8;
			    //cout<<"ellipse_flag: "<<flag[0]<<flag[1]<<flag[2]<<flag[3]<<" "<<MAX(box.size.width, box.size.height)/MIN(box.size.width, box.size.height)<<endl;
			    if( flag[0]|| flag[1]|| flag[2]||flag[3])  
			      continue;  
			    possible_cirle.push_back(box);    
			    id_of_cirlce.push_back(i);   
			    contours_id.push_back(i);
			    //»æÖÆÍÖÔ²  <---------------ALL
			    ellipse(cimage, box, Scalar(25,0,255), 1.68, CV_AA);  
			    
			    //imshow("ra",cimage);
			    }
			    
		      }
		   #endif
		  
		  #endif
      #else//circle with circle head
 	      Mat  test;
	      img_gray.copyTo(test);
	      vector< Point2f> v_use;
 	      for(int k=0;k<0.98*count;k++)
	          v_use.push_back( Point2f(pointsf.at<float>(k,0),pointsf.at<float>(k,1)) );
	       if(v_use.size()>10){ 
			RotatedRect box = fitEllipse(v_use);    
			int  flag[6];
			flag[0]=MAX(box.size.width, box.size.height)/MIN(box.size.width, box.size.height)>3;
			flag[1]=box.size.width* box.size.height <img_gray.cols*img_gray.rows*(float)min_box/1000.;
			flag[2]=box.size.width* box.size.height >img_gray.cols*img_gray.rows*0.8;
			flag[3]=MAX(box.size.width, box.size.height) > MAX(img_gray.cols,img_gray.rows)*0.8;
			//cout<<"ellipse_flag: "<<flag[0]<<flag[1]<<flag[2]<<flag[3]<<" "<<MAX(box.size.width, box.size.height)/MIN(box.size.width, box.size.height)<<endl;
			if(!( flag[0]|| flag[1]|| flag[2]||flag[3])){
			    possible_cirle.push_back(box);    
			    id_of_cirlce.push_back(i);   
			    contours_id.push_back(i);
			    ellipse(cimage, box, Scalar(25,0,255), 1.68, CV_AA);  
			}
	      }
      #endif
    }  

    // circle _ detect   by circle  match
   if(possible_cirle.size()>=2&&1)
   {
     //cout<<"possible_circle:"<< possible_cirle.size()<<endl;
     float center_ero[possible_cirle.size()][possible_cirle.size()];
     float k_lm_ero[possible_cirle.size()][possible_cirle.size()];
     float k_ero[possible_cirle.size()][possible_cirle.size()];
     float  angle_ero[possible_cirle.size()][possible_cirle.size()];
     float  all_ero[possible_cirle.size()][possible_cirle.size()];
     float max_cero=0;
      for(int i=0;i<possible_cirle.size();i++)
      {
	  for(int j=0;j<possible_cirle.size();j++)
	  if(i!=j){
	    float  cx1=possible_cirle[i].center.x;
	    float  cy1=possible_cirle[i].center.y;
	    float  cx2=possible_cirle[j].center.x;
	    float  cy2=possible_cirle[j].center.y;
	    float center_ero=sqrt( pow(cx1-cx2,2)+pow(cy1-cy2,2) ) ;
             if(max_cero<center_ero)
	       max_cero=center_ero;
	  }
      }
      for(int i=0;i<possible_cirle.size();i++)
      {
	  for(int j=0;j<possible_cirle.size();j++)
	  if(i!=j){
	    float  cx1=possible_cirle[i].center.x;
	    float  cy1=possible_cirle[i].center.y;
	    float  cx2=possible_cirle[j].center.x;
	    float  cy2=possible_cirle[j].center.y;
	    float   w1=possible_cirle[i].size.width;
	    float   h1=possible_cirle[i].size.height;
	    float   w2=possible_cirle[j].size.width;
	    float   h2=possible_cirle[j].size.height;
	    if(possible_cirle.size()>2)
	    center_ero[i][j]=sqrt( pow(cx1-cx2,2)+pow(cy1-cy2,2) )/max_cero ;
	    k_lm_ero[i][j]=abs( 1-((w1/h1)/(w2/h2)));
	    k_ero[i][j]=      abs( 1- k_circle/ (MAX(w1,w2)/MIN(w1,w2)) );
	    angle_ero[i][j]=abs(possible_cirle[i].angle-possible_cirle[j].angle)/360;
	    all_ero[i][j]=angle_ero[i][j]*1+k_lm_ero[i][j]+center_ero[i][j]+k_ero[i][j];
	  //cout<<i<<j<<"  ce:"<<center_ero[i][j]<<"   ae:"<<angle_ero[i][j]<<"  ke:"<<k_lm_ero[i][j]<<"  ker:"<<k_ero[i][j]<<"  ale:"<< all_ero[i][j]<<endl;
	  }
      } 
	 float min_err=100;
	 int no[2]={0,0};
	 for(int i=0;i<possible_cirle.size();i++)
        {
	  for(int j=0;j<possible_cirle.size();j++){
	         if(i!=j){
		     int  g_dsrcAreai =possible_cirle[i].size.width*possible_cirle[i].size.height;
		     int  g_dsrcAreaj =possible_cirle[j].size.width*possible_cirle[j].size.height;
		     float  cx1=possible_cirle[i].center.x;
		     float  cy1=possible_cirle[i].center.y;
		     float  cx2=possible_cirle[j].center.x;
		     float  cy2=possible_cirle[j].center.y;
		     int flag[6];
		     flag[0]=center_ero[i][j]<0.99;//<<TODO;
		     flag[1]=angle_ero[i][j]<0.4  ;
		     flag[2]=k_ero[i][j]>0.0001;
		     flag[3]=1;//abs(g_dsrcAreai-g_dsrcAreaj)>cr2/cr1*0.2*MAX(g_dsrcAreai,g_dsrcAreaj);
		     flag[4]=sqrt( pow(cx1-cx2,2)+pow(cy1-cy2,2) )<MIN(bimage.cols,bimage.rows)*0.06;
		   // cout<<i<<j<<" "<<abs(g_dsrcAreai-g_dsrcAreaj)<<" "<<MAX(g_dsrcAreai,g_dsrcAreaj)<<endl;
		     //cout<<i<<j<<"  ce:"<<center_ero[i][j]<<"   ae:"<<angle_ero[i][j]<<"  ke:"<<k_lm_ero[i][j]<<"  ker:"<<k_ero[i][j]<<"  ale:"<< all_ero[i][j]<<endl;
		    
		  //kick two circle is two near
		  int flag_cross=0;
		  if(abs(g_dsrcAreai-g_dsrcAreaj)<100)
			flag_cross=1;
		 // cout<<id_of_cirlce[i]<<" "<<id_of_cirlce[j]<<" "<<center_ero[i][j]<<" "<<flag[0]<<flag[1]<<flag[2]<<flag[3]<<flag[4]<<flag_cross<<endl;
		  #if HEAD_GAP
		  if(id_of_cirlce[i]==id_of_cirlce[j]&&min_err>all_ero[i][j]&&flag[0]&&flag[1]&&flag[2]&&flag[3]&&flag[4]&&flag_cross==0)
		  #else
		  if(id_of_cirlce[i]!=id_of_cirlce[j]&&min_err>all_ero[i][j]&&flag[0]&&flag[1]&&flag[2]&&flag[3]&&flag[4]&&flag_cross==0)		      
		  #endif
		  {
		      min_err=all_ero[i][j];
		      no[0]=i;no[1]=j;
		  }
		}
	     }
	 }
	 int test=0;
	//no[0]=3;no[1]=0; test=1;
	//cout<<"e_real:"<<no[0]<<no[1]<<" "<<all_ero[no[0]][no[1]]<<endl;
	if((all_ero[no[0]][no[1]]<1.2  &&(no[0]+no[1]>0))||test){
	      //cout<<"Circle_check!!!!!!!!!!!!!!!!!!!:"<<no[0]<<no[1]<<" "<<all_ero[no[0]][no[1]]<<endl;
	      circle_flag=1;
	      int id_o,id_i;
	      if(possible_cirle[no[0]].size.height>possible_cirle[no[1]].size.height)
	      {
		id_o=no[0];id_i=no[1];
		}else{
		id_o=no[1];id_i=no[0];
		}
	      //cout<<no[0]<<no[1]<<endl;
	      circle_found.push_back(possible_cirle[id_o]); circle_found.push_back(possible_cirle[id_i]);
	      int  line_w=MAX(1,abs( MIN( abs(possible_cirle[id_o].size.height-possible_cirle[id_i].size.height),
					  abs(possible_cirle[id_o].size.width-possible_cirle[id_i].size.width)))*(float)d_wide/100.);
	      ellipse(white_img, possible_cirle[id_o], Scalar(255,255,255), line_w, CV_AA);  circle(cimage,possible_cirle[id_o].center,4,Scalar(255),4);
	      ellipse(white_img, possible_cirle[id_i], Scalar(255,255,255),line_w, CV_AA);  circle(cimage,possible_cirle[id_i].center,1,Scalar(255),1);	      
	      ellipse(cimage, possible_cirle[id_o], Scalar(0,255,255), 4, CV_AA);  circle(cimage,possible_cirle[id_o].center,4,Scalar(255),4);
	      ellipse(cimage, possible_cirle[id_i], Scalar(255,0,255),2, CV_AA);  circle(cimage,possible_cirle[id_i].center,1,Scalar(255),1);	      
	 
	      drawContours(white_img, contours, contours_id[id_o], Scalar::all(255), CV_FILLED);
	      drawContours(white_img, contours, contours_id[id_i], Scalar::all(255), CV_FILLED);
	           
	      drawContours(white_img2, contours, contours_id[id_o], Scalar(0,0,0), CV_FILLED);
	      drawContours(white_img2, contours, contours_id[id_i], Scalar::all(255), CV_FILLED);
	      img_mask(img_rgb,white_img1,white_img2);
           
	     //imshow("sasd",white_img2);
	 }           
      }

 // -----------------------check the head gap and decided wheather it is a circle --------------------------------
  Point Center,Front;
   #if HEAD_GAP
  if(circle_flag==1 && 1)
  {
        vector<Point2f>corners;
	Mat gray_circle=Mat::zeros(white_img.size(),CV_8UC1);
	cvtColor(white_img,gray_circle,CV_BGR2GRAY);
	double qualityLevel = 0.5;
	double minDistance = 3;
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;
	goodFeaturesToTrack(gray_circle,corners,6,qualityLevel,minDistance,Mat(),blockSize,useHarrisDetector,k);
	Size winSize = Size(5, 5);
	Size zeroZone = Size(-1, -1);
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001);
	//if(corners.size())
	//cornerSubPix(gray_circle, corners, winSize, zeroZone, criteria);
	//for(int i=0;i<corners.size();i++)
	//{
		//circle(cimage,corners[i],2,Scalar(255,0,60),2);	
	//}
     
        if(corners.size()>2||1)
	{
	      int cx=circle_found[0].center.x/2+circle_found[1].center.x/2;
	      int cy=circle_found[0].center.y/2+circle_found[1].center.y/2;
	     // cout<<"test"<<endl;
	      Center.x=cx;Center.y=cy;
	    // circle(cimage,Point2d(cx,cy),3,Scalar(255),3);	
	      Mat white_imgg;
	      vector<vector<Point>> all_contours,contours1;
	      cvtColor(white_img,white_imgg,CV_BGR2GRAY);
	      //Mat element = getStructuringElement(MORPH_RECT, Size(  1,1 ));
	      //erode(white_imgg, white_imgg, element);
	      //Mat element2 = getStructuringElement(MORPH_RECT, Size(3,3 ));
	      //dilate(white_imgg, white_imgg, element2);
	  //  imshow("e",white_imgg);
	    findContours(white_imgg, all_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
          
	    for (int i = 0; i < all_contours.size(); ++i)
		    if (all_contours[i].size() > 10)
			    contours1.push_back(all_contours[i]);
		    
	    vector<Point> approx_poly;
	    vector<Rect>rect(contours1.size());
	    int min_area=1000000;
	    int max_area=0;
	    int frone_contours_id=99;
	    for (int i = 0; i < contours1.size(); ++i)
	    {
		    double eps = contours1[i].size()*0.1;
		    approxPolyDP(contours1[i], approx_poly, eps, true);
		    Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));  
		  	 
		    if (approx_poly.size() > 4)
			    continue;	
		    if (!isContourConvex(approx_poly))
			    continue;
		    //drawContours(cimage, contours1, i, color,1, 4, vector<Vec4i>(), 0, Point());  
		   //  imshow("cimage",cimage);
		    Rect temp= boundingRect(contours1[i]);
		    int  g_dsrcArea =temp.width*temp.height;
		    int s1= g_dsrcArea;//contours1[i].size();
		    int s2= circle_found[1].size.width* circle_found[1].size.height*0.618;
		    int flag[4]={0};
		    flag[0]=min_area>g_dsrcArea;
		    flag[1]=s1<s2;
		    flag[2]=g_dsrcArea>s2*0.0334;
		   // cout<<"g_dsrcArea: "<<g_dsrcArea<<endl;
		   // cout<< flag[0]<< flag[1]<<flag[2]<<" "<<min_area<<" "<<s1 <<" "<<s2<<endl;
		    if(flag[0]&& flag[1]&&flag[2])
		    {frone_contours_id=i;min_area=g_dsrcArea;}
	    }
           // cout<<"frone_contours_id:  "<<frone_contours_id<<endl;
	    if(frone_contours_id!=99){
	    center_circle.x=Center.x;
	    center_circle.y=Center.y;
	    //cout<<"Circle_Tag_Dected!!!!!!!!!!!!!!"<<endl;
	    Rect temp;
	    temp=boundingRect(contours1[frone_contours_id]);
	    int x=temp.x;
	    int y=temp.y;
	    int width=temp.width;
	    int height=temp.height;
	    Front.x=x+temp.width/2;Front.y=y+temp.height/2;
	    //rectangle(cimage,Point(x,y),Point(x+width,y+height),Scalar(0,0,255),1);
	    //circle(cimage,Point(Front.x,Front.y),2,Scalar(25,25,255),2);	
	    int dis_c=sqrt(pow(Front.x-Center.x,2)+pow(Front.y-Center.y,2)) ;
	   // cout<<dis_c<<endl;
	    if(dis_c >15)
	    	circle_flag=2;
	    
	    if(circle_flag==2 && 0){//cal front tangle
	          //cout<<"test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
		  float yaw=atan2(Front.x-Center.x,Front.y-Center.y)*57.3;
		  int color_tag_check=0,cnt_insert[3]={0};
		  int front[10]={0},back[10]={0},middle[10]={0},left[10]={0},right[10]={0};
		  Point EP[2],Left,Right;
		  EP[0].x=center_of_tag_roi[0];  EP[0].y=center_of_tag_roi[1];
		  EP[1].x=center_of_tag_roi[2];  EP[1].y=center_of_tag_roi[3];
		  //cout<<"Yaw:"<<yaw<<endl;
		  //left right divide
		        for(int i=0;i<2;i++)
		       {
			      if(check_point_front_arrow(EP[i].x,EP[i].y,Center.x,Center.y,yaw+90)){
			      circle(cimage,Point2f(EP[i].x,EP[i].y),4,Scalar(0,0,255),4);
			      left[i]=1;
			      Left=EP[i];
			      }
			    else 
			    {
			      right[i]=1;
			      circle(cimage,Point2f(EP[i].x,EP[i].y),6,Scalar(0,0,255),6); 
			        Right=EP[i];
			    }
	               }
	           /*/cal ambigous angel
	            Mat rot_vec, tvec;
		    vector<Point3f> pts_3d;
		    vector<Point2f> pts_2d;
		    pts_3d.push_back ( Point3f ( 0,0,0 ) );
		    pts_3d.push_back ( Point3f ( 0,(cr1+cr2)/2,0 ) );//f
		    pts_3d.push_back ( Point3f (- (cr1+cr2)/2*cos(45*0.0173), (cr1+cr2)/2*sin(45*0.0173),0 ) );//l
		    pts_3d.push_back ( Point3f (  (cr1+cr2)/2*cos(45*0.0173), (cr1+cr2)/2*sin(45*0.0173),0 ) );//r
		    pts_2d.push_back ( Point2f(Center.x,Center.y) );
		    pts_2d.push_back ( Point2f(Front.x,Front.y) );
		    pts_2d.push_back ( Point2f(Left.x,Left.y) );
		    pts_2d.push_back ( Point2f(Right.x,Right.y) );
		    // cout << "World Point= " << endl << pts_3d << endl;cout << "Figure Point= " << endl << pts_2d<< endl;
		    //bool res = solvePnP(pts_3d, pts_2d, camera_matrix, distortion_coefficients, rot_vec, tvec);
	
		    //getCameraPos(rot_vec,tvec,pos_c,acc_c);*/
		    //cout<<"Circle_AM:"<<pos_c.x<<" "<<pos_c.y<<" "<<pos_c.z<<" ATT::"<<acc_c.x<<" "<<acc_c.y<<" "<<acc_c.z<<endl;
	            circle_flag=3;    
	    }
	}
	}
  }
  #else//head check for circle head


  #endif
  
 // --------------------------------------------use circle information to calculate the position---------------------------------------------
  if(circle_flag>=2&&1)//cal figure point
  {//  Point Center,Front;//circle_found O->I
    //outer c
     float theta = circle_found[0].angle * CV_PI / 180.0 ;  
     float m1 = circle_found[0].size.width/4;  
     float m2 = circle_found[0].size.height/4;  
     float circley = circle_found[0].center.x ;//+ col_range.start; // x/y appear inverted
     float circlex = circle_found[0].center.y ;//+ row_range.start;
     float circlem0 = circle_found[0].size.width * 0.25;
     float circlem1 = circle_found[0].size.height * 0.25;
     float circlev0 = cos(circle_found[0].angle / 180.0 * M_PI);
     float circlev1 = sin(circle_found[0].angle / 180.0 * M_PI);
     double x,y,x1,y1,x2,y2;
      //transform the center
      transform1((double)Center.x,(double)Center.y, x, y);

      //calculate the major axis 
      //endpoints in image coords
      double sx1 = circlex + circlev0 * circlem0 * 2;
      double sx2 = circlex - circlev0 * circlem0 * 2;
      double sy1 = circley + circlev1 * circlem0 * 2;
      double sy2 = circley - circlev1 * circlem0 * 2;
      // circle(cimage,Point2f(sx1,sy1),2,Scalar(220,220,222),2);	circle(cimage,Point2f(sx2,sy2),4,Scalar(220,220,222),2);
      //endpoints in camera coords 
      transform1(sx1, sy1, x1, y1);
      transform1(sx2, sy2, x2, y2);
      //semiaxis length 
      float major = sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))/2.0;

      float  v0 = (x2-x1)/major/2.0;
      float  v1 = (y2-y1)/major/2.0;

      //calculate the minor axis 
      //endpoints in image coords
      sx1 = circlex + circlev1 * circlem1 * 2;
      sx2 = circlex - circlev1 * circlem1 * 2;
      sy1 = circley - circlev0 * circlem1 * 2;
      sy2 = circley + circlev0 * circlem1 * 2;
      //endpoints in camera coords 
      transform1(sx1, sy1, x1, y1);
      transform1(sx2, sy2, x2, y2);

      //semiaxis length 
      float minor = sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))/2.0;
     
     
	double a,b,c,d,e,f;
	a = v0*v0/(major*major)+v1*v1/(minor*minor);
	b = v0*v1*(1/(major*major)-1/(minor*minor));
	c = v0*v0/(minor*minor)+v1*v1/(major*major);
	d = (-x*a-b*y);
	e = (-y*c-b*x);
	f = (a*x*x+c*y*y+2*b*x*y-1);
	cv::Matx33d data(a,b,d,
									 b,c,e,
									 d,e,f);
	cv::Vec3d eigenvalues;
	cv::Matx33d eigenvectors;
	cv::eigen(data, eigenvalues, eigenvectors);
	double L1 = eigenvalues(1);
	double L2 = eigenvalues(0);
	double L3 = eigenvalues(2);
	int V2 = 0;
	int V3 = 2;
	int V1=1;
	// position
	Vec3f pos,rot,posm,rotm,posmt;
	int S1=-1,S2=1,S3=1;//-1 1 s1
	// rotation
	Vec3f rots[4];
	cv::Matx13d normal_mat;
	Vec3f romtmm;

        stringstream ss,ss1,ss2,ss3;
	float  yaw=To_180_degrees(180-atan2(Front.x-Center.x,Front.y-Center.y)*57.3);
       // cout<<"YAW_CIRCLE:"<<yaw<<"-----------------------------------"<<endl;
	romtmm(2)=yaw;
	 ss2 <<romtmm;
	// putText(cimage, ss2.str(), Point(2,66), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,0,255),2);
        S1=1;S2=1;
        float circle_diameter=cr1*2;
	cv::Matx13d position_mat =S1* L3 * sqrt((L2 - L1) / (L2 - L3)) * eigenvectors.row(V2)
	+ S2*L2 * sqrt((L1 - L3) / (L2 - L3)) * eigenvectors.row(V3);
	pos = cv::Vec3f(position_mat(0), position_mat(1), position_mat(2));
	double z =circle_diameter/sqrt(-L2*L3)/2.0;
	S3 = (pos(2) * z < 0 ? -1 : 1);	
	pos *= S3 * z;

          //----------------------------------------try3----------------------------------------------
	 V3=0;
	 V2=2;
	 V1=1;
          //eigenvalues
	 L1 = eigenvalues(V1); 
	 L2 = eigenvalues(V2);
	 L3 = eigenvalues(V3);
	//eigenvectors
	//detected pattern position
	float rd = circle_diameter/sqrt(-L2*L3)/2.0;
	float c0 =  sqrt((L2-L1)/(L2-L3));
	float c0x = c0* eigenvectors(V2,0);//V[2][V2];
	float c0y = c0*eigenvectors(V2,1);//V[1][V2];
	float c0z = c0*eigenvectors(V2,2);//V[2][V2];
	float c1 =  sqrt((L1-L3)/(L2-L3));
	float c1x = c1*eigenvectors(V3,0);//V[0][V3];
	float c1y = c1*eigenvectors(V3,1);//V[1][V3];
	float c1z = c1*eigenvectors(V3,2);//V[2][V3];

	float z0 = -L3*c0x+L2*c1x;
	float z1 = -L3*c0y+L2*c1y;
	float z2 = -L3*c0z+L2*c1z;
	if (z2*rd < 0){ z2 = -z2;z1 = -z1;z0 = -z0;
}
	x = z0*rd;	y = z1*rd;z = z2*rd;
	float xx1=x,yy1=y,zz1=z;
	float s1,s2;
	//select
	stringstream  uvss[4];	 stringstream ss4,ss5,ss6;
	Point2f uv_p[4]; Point3f  err[4];
	vector <Point3f> att_reg,pos_reg;
	//1 -1  0----------------------------------------------------------------------------------------------
	s1=1;s2=-1;
	float n1 = +s1*c0x+s2*c1x;
	float n0 = +s1*c0y+s2*c1y;
	float n2 = +s1*c0z+s2*c1z;//cout<<n2*57.3<<endl;
	n2=yaw/57.3;
	Point3f Frone_3d=Point3f(0,-(cr1+cr2)/2,0);

	// n1=rots[2][1];n1=rots[2][0];
	z0 = s1*L3*c0x+s2*L2*c1x;
	z1 = s1*L3*c0y+s2*L2*c1y;
	z2 = s1*L3*c0z+s2*L2*c1z;
	if (z2*rd < 0){ z2 = -z2;z1 = -z1;z0 = -z0;
		  n0*=-1;n1*=-1;
	}
	x = z0*rd;	y = z1*rd;z = z2*rd;
       //  x=xx1;y=yy1;z=zz1;
	x=pos(0);y=pos(1);z=pos(2);
	 att_reg.push_back(Point3f(n0*57.3,n1*57.3,n2*57.3)); pos_reg.push_back(Point3f(x,y,z));
	// ProjectCheck(Vec3f(n0*57.3,n1*57.3,n2*57.3), Vec3f(x,y,z), Frone_3d, Front ,uv_p[0],err[0]);
	// cout<<"uv_p"<<uv_p[0]<<endl;
	// cout<<Vec3f(x,y,z)<<endl;
	 uvss[0]<<"0";	 ss4 <<n0*57.3<<" "<<n1*57.3<<"  "<<err[0].z;
	 //putText(cimage, uvss[0].str(), uv_p[0], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,125,125),2);
	 //1 1  2----------------------------------------------------------------------------------------------
	s1=1;s2=1;
	n1 = +s1*c0x+s2*c1x;
	n0 = +s1*c0y+s2*c1y;
	n2 = +s1*c0z+s2*c1z;//cout<<n2*57.3<<endl;
	n2=yaw/57.3;

	z0 = s1*L3*c0x+s2*L2*c1x;
	z1 = s1*L3*c0y+s2*L2*c1y;
	z2 = s1*L3*c0z+s2*L2*c1z;
	if (z2*rd < 0){ z2 = -z2;z1 = -z1;z0 = -z0; 
	n0*=-1;n1*=-1;
	}
	x = z0*rd;	y = z1*rd;z = z2*rd;
	// x=xx1;y=yy1;z=zz1;
	 x=pos(0);y=pos(1);z=pos(2);
	 att_reg.push_back(Point3f(n0*57.3,n1*57.3,n2*57.3));pos_reg.push_back(Point3f(x,y,z));
	//  ProjectCheck(Vec3f(n0*57.3,n1*57.3,n2*57.3), Vec3f(x,y,z),Frone_3d,Front ,uv_p[2],err[2]);
	// cout<<"uv_p"<<uv_p[2]<<endl;
	  //	 cout<<Vec3f(x,y,z)<<endl;
	 uvss[2]<<"2";	ss5 <<n0*57.3<<" "<<n1*57.3<<"  "<<err[2].z;
	 //putText(cimage, uvss[2].str(), uv_p[2], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,255,125),2);
	//--------------------------------------------------------------------------------------------------------------------------------//
	// -----------------------------------------decide  which is the correct result-------------------------------------------
	int min_id=99;
	att_reg[0].x*=-1;
	att_reg[1].x*=-1;
	#if !USE_ARUCO
	if(m_markers.size()&&1)
	{
	   float  dis[4];
	   dis[0]=abs(att_2d.x-att_reg[0].x);
	   dis[1]=abs(att_2d.y-att_reg[0].y);
	   dis[2]=abs(att_2d.x-att_reg[1].x);
	   dis[3]=abs(att_2d.y-att_reg[1].y);
	  // cout<<att_2d<<" 2C(1): " <<att_reg[0]<<" (2)"<<att_reg[2]<<endl;
	   if(dis[0]+dis[1]<dis[2]+dis[3])   
	     min_id=0;
	   else
	     min_id=1;
	}
	#else //ARUCO Version
	if(aruco_size&&1)
	{
	   float  dis[4];
	   float temp[2];
	   temp[0]=  -(cos(-att_2d.z*0.0173)*att_2d.x -sin(-att_2d.z*0.0173)*att_2d.y);
	   temp[1]=   (sin(-att_2d.z*0.0173)*att_2d.x +cos(-att_2d.z*0.0173)*att_2d.y);
	   dis[0]=abs(temp[0]-att_reg[0].x);
	   dis[1]=abs(temp[0]-att_reg[0].y);
	   dis[2]=abs(temp[1]-att_reg[1].x);
	   dis[3]=abs(temp[1]-att_reg[1].y);
	   //cout<<temp[0]<<" "<<temp[1]<<" 2C(1): " <<att_reg[0]<<" (2)"<<att_reg[1]<<endl;
	   if(dis[0]+dis[1]<dis[2]+dis[3])   
	     min_id=0;
	   else
	     min_id=1;
	}
	#endif
	else if(circle_flag==3 &&0)
	{
	    float  dis[4];
	   dis[0]=abs(acc_c.x-att_reg[0].x);
	   dis[1]=abs(acc_c.y-att_reg[0].y);
	   dis[2]=abs(acc_c.x-att_reg[1].x);
	   dis[3]=abs(acc_c.y-att_reg[1].y);
	   //cout<<"CC:  "<<acc_c<<" ?: "<<att_reg[0]<<" "<<att_reg[1]<<"  DIS:  "<<dis[0]+dis[1]<<" "<<dis[2]+dis[3]<<endl;
	   if(dis[0]+dis[1]<dis[2]+dis[3])   
	     min_id=0;
	   else
	     min_id=1;
	}else if( 1    )//imu
	{
  	   float  dis[4];
	   float temp[2];
	   //float att_drone[3];
	   float yaw_off=att_reg[1].z-att_drone[2];
	   temp[0]=att_drone[0]-45;
	   temp[1]=att_drone[1];
	   dis[0]=abs(temp[0]-att_reg[0].x);
	   dis[1]=abs(temp[1]-att_reg[0].y);
	   dis[2]=abs(temp[0]-att_reg[1].x);
	   dis[3]=abs(temp[1]-att_reg[1].y);
	   //cout<<"IMU: "<< temp[0]<<" "<<temp[1]<<" ?(1)): " <<att_reg[0]<<" (2): "<<att_reg[1]<<endl;
	   if(dis[0]+dis[1]<dis[2]+dis[3])   
	     min_id=0;
	   else
	     min_id=1;
	}
	/* ----------------------------------------------------connerl corrdinate--------------------------------------------------------
	 * 
	 *     x-       z+   
	 * 
	 *                y+
	 */
	if(min_id!=99){
	att_reg[min_id].z*=-1;
	//putText(cimage, ss6.str(), Point(2,460), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,255,125),2);putText(cimage, ss4.str(), Point(2,400), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(125,125,125),2); putText(cimage, ss5.str(), Point(2,420), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(125,125,125),2);
        float angle[3]={-att_reg[min_id].y,       att_reg[min_id].x,           att_reg[min_id].z};
	Matrix3d   trans1,trans2,trans3;  
	trans1<< 1,0,0, 
	0,cos(angle[0]*0.0173),-sin(angle[0]*0.0173), 
	0,sin(angle[0]*0.0173),cos(angle[0]*0.0173);
	trans2<<
	cos(angle[1]*0.0173),0,sin(angle[1]*0.0173),
	0,1,0, 
	-sin(angle[1]*0.0173),0,cos(angle[1]*0.0173);
	trans3<<
	cos(angle[2]*0.0173), -sin(angle[2]*0.0173),0, 
	sin(angle[2]*0.0173),   cos(angle[2]*0.0173), 0,
	0,0,1;
	Vector3d tt(pos_reg[min_id].x,-pos_reg[min_id].y,pos_reg[min_id].z);  
	Vector3d o2=trans2*trans1*tt;
	float dis=sqrt(pow(tt[0],2)+pow(tt[1],2)+pow(tt[2],2));
	Vector3d o3;

	posmt(0)=o2(1);posmt(1)=o2(2);posmt(2)=o2(0);
	Vector3d t1(o2(1),o2(2),-o2(0));  
	o3=trans3*t1;
	//float gain_length[2]={0.45,1.7};
	float gain_length[2]={1,1};
	posmt(0)=o3(0)*gain_length[0];posmt(1)=-o3(1)*gain_length[0];posmt(2)=o3(2)*gain_length[1];
       	//cout<<"angle: "<<att_reg[min_id]<<" Origin: "<<tt[0]<<"  "<<tt[1]<<"  "<<tt[2]<<"  Result: "<<posmt<<endl; 
	    if(posmt(2)>0||0 ){//kick the abnormal z position result
	    m_markers_circle.push_back(circle_found[0]); m_markers_circle.push_back(circle_found[1]);
	    m_circle_center=Center;
	    m_circle_head=Front;
	    pos_circle=Point3f(posmt(0),posmt(1),posmt(2));
	    att_circle=att_reg[min_id];
	    }
	}
  }
   // imshow("white",white_img);
    //imshow("result", cimage); 
}

void MarkerRecognizer::bubbleSort(float* pData, int * id,int length)
{
    float temp;
    int  temp_id;
    for(int i = 0;i != length;++i)
    {
        for (int j = 0; j != length; ++j)
        {
            if (pData[i] < pData[j])
            {
                temp = pData[i];
                pData[i] = pData[j];
                pData[j] = temp;
		temp_id = id[i];
		id[i] = id[j];
                id[j] = temp_id;
            }
        }
    }
}

void MarkerRecognizer::ProjectCheck(Vec3f rot,Vec3f pos, Point3f pos3d, Point2f uv,  Point2f &uv_p, Point3f & err)
{
        float angle[3]={rot(0),rot(1),rot(2)};
	Matrix3d   trans1;  
	trans1<< 1,0,0, 
	0,cos(angle[0]*0.0173),-sin(angle[0]*0.0173), 
	0,sin(angle[0]*0.0173),cos(angle[0]*0.0173);
	Matrix3d  trans2;  
	trans2<<cos(angle[1]*0.0173),0,sin(angle[1]*0.0173),
	0,1,0, 
	-sin(angle[1]*0.0173),0,cos(angle[1]*0.0173);
	Matrix3d  trans3;  
	trans3<<cos(angle[2]*0.0173), -sin(angle[2]*0.0173),0, 
	sin(angle[2]*0.0173),cos(angle[2]*0.0173),0,
	0,0,1;
	Matrix3d o2=trans1*trans2*trans3;
	//cout<<o2<<endl;
	Mat R = (cv::Mat_<float>(3,3) <<
	         o2(0,0),o2(0,1),o2(0,2),
	         o2(1,0),o2(1,1),o2(1,2),
		 o2(2,0),o2(2,1),o2(2,2)
	);
	Mat r;
	cv::Rodrigues (R, r); 
	vector<cv::Point2f> reProjection;
	std::vector<cv::Point3f> totalPre;
	totalPre.push_back(pos3d);
	Mat rvec =r;
	Mat tvec = (cv::Mat_<float>(3,1) <<
	        pos(0),pos(1),pos(2)
	);
        // cout<<" "<<tvec<<endl;
       //  cout<<" "<<rvec<<endl;
	cv::projectPoints(totalPre,rvec,tvec,camera_matrix, distortion_coefficients,reProjection);
	//cout<<" "<<reProjection[0]<<endl;
	uv_p=reProjection[0];
	err.x=(uv_p.x-uv.x);
	err.y=(uv_p.y-uv.y);
	err.z=sqrt(err.x*err.x+err.y*err.y);
	 //  cout<<uv_p<<"  "<<uv<<endl;
	 // cout<<"Err:"<<err<<endl;
}

void MarkerRecognizer::RodriguesR(cv::Matx13d normal_mat,Vec3f &rot){
	Mat R(3, 3, CV_32FC1);
	cv::Rodrigues (normal_mat, R ); // rÎªÐý×ªÏòÁ¿ÐÎÊœ£¬ÓÃRodrigues¹«Êœ×ª»»ÎªŸØÕó
	double r11 = R.ptr<double>(0)[0];
	double r12 = R.ptr<double>(0)[1];
	double r13 = R.ptr<double>(0)[2];
	double r21 = R.ptr<double>(1)[0];
	double r22 = R.ptr<double>(1)[1];
	double r23 = R.ptr<double>(1)[2];
	double r31 = R.ptr<double>(2)[0];
	double r32 = R.ptr<double>(2)[1];
	double r33 = R.ptr<double>(2)[2];
	float E1,E2,E3,dlta,pi=3.1415926;
	if (r13 == 1 ||r13== -1){
	E3 = 0; 
	dlta = atan2(r12,r13);
	if( r13== -1){
	E2 = pi/2;
	E1 = E3 + dlta;}
	else{
	E2 = -pi/2;
	E1 = -E3 + dlta;}
	}
	else{
	E2 = - asin(r13);
	E1 = atan2(r23/cos(E2), r33/cos(E2));
	E3 = atan2(r12/cos(E2), r11/cos(E2));
	}
	rot(0)=To_180_degrees(E1*57.3);rot(1)=E2*57.3;rot(2)=-E3*57.3;
}

void MarkerRecognizer::transform1(double x_in, double y_in, double& x_out, double& y_out) 
{
  //#if defined(ENABLE_FULL_UNDISTORT)
 x_out = (x_in-camera_matrix.ptr<double>(0)[2] )/camera_matrix.ptr<double>(0)[0];
 y_out = (y_in-camera_matrix.ptr<double>(1)[2] )/camera_matrix.ptr<double>(1)[1];
 // #else
  std::vector<cv::Vec2d> src(1, cv::Vec2d(x_in, y_in));
  std::vector<cv::Vec2d> dst(1);
   cv::undistortPoints(src, dst, camera_matrix, distortion_coefficients);
  x_out = dst[0](0); y_out = dst[0](1);
 // #endif
}

//---------------------------------------------------------------------
void MarkerRecognizer::markerDetect(Mat& img_rgb, vector<Marker1>& possible_markers, int min_size, int min_side_length)
{ 
  
	Mat img_bin;
	Mat img_gray;
	 static int init=0;
   int  good[2]={0};
   static int TH=255;
   static int TL=50;
   static int TT=10;
    static int APPROX_POLY_EPS1=50;
     static int min_sizer=150;
   if(!init && 0)
   {
     namedWindow("contouers", CV_WINDOW_AUTOSIZE); //create a window called "Control"
    //Create trackbars in "Control" window
    cvCreateTrackbar("TH", "contouers", &TH, 255); //Hue (0 - 179)
    cvCreateTrackbar("TL", "contouers", &TL, 255); //Hue (0 - 179)
    cvCreateTrackbar("TT", "contouers", &TT, 50); //Hue (0 - 179)
    cvCreateTrackbar("min_size", "contouers", &min_sizer, 255); //Hue (0 - 179)
    cvCreateTrackbar("min_EPS1", "contouers", &APPROX_POLY_EPS1, 1000); //Hue (0 - 179)
    init =1;
  }
  
        cvtColor(img_rgb, img_gray, CV_BGRA2GRAY);
	  
	int thresh_size = (min_sizer/4)*2 + 1;
	adaptiveThreshold(img_gray, img_bin, TH, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, thresh_size, thresh_size/3);
	//threshold(img_gray, img_bin, TL, TH, THRESH_BINARY_INV|THRESH_OTSU);
	//AdaptiveThereshold(img_gray, img_bin,TT );
	//imshow("3",img_bin);
	//morphologyEx(img_bin, img_bin, MORPH_OPEN, Mat());	//use open operator to eliminate small patch
        //imshow("2",img_bin);
	vector<vector<Point>> all_contours;
	vector<vector<Point>> contours;
	findContours(img_bin, all_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	for (int i = 0; i < all_contours.size(); ++i)
	{
		if (all_contours[i].size() > min_size)
		{
			contours.push_back(all_contours[i]);
		}
	}
	Mat drawing = Mat::zeros(img_bin.size(), CV_8UC3);  
        //cout<<"Csize:="<<contours.size()<<endl;
	int flag[20]={0};
	vector<Point> approx_poly;
	for (int i = 0; i < contours.size(); ++i)
	{
		
		double eps = contours[i].size()*(float)APPROX_POLY_EPS1/1000.;
		approxPolyDP(contours[i], approx_poly, eps, true);
		 Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));  
		if (approx_poly.size() != 4)
			continue;	
       		if (!isContourConvex(approx_poly))
			continue;
	       // drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());  
		
		//Ensure that the distance between consecutive points is large enough
		float min_side = FLT_MAX;
		for (int j = 0; j < 4; ++j)
		{
			Point side = approx_poly[j] - approx_poly[(j+1)%4];
			min_side = min(min_size, side.dot(side));
		}
		//cout<<"min_side:"<<min_side<<endl;
		//Sort the points in anti-clockwise
		Marker1 marker = Marker1(0, approx_poly[0], approx_poly[1], approx_poly[2], approx_poly[3]);
		int dis[4];
		dis[0]=sqrt( pow(marker.m_corners[0].x-marker.m_corners[1].x,2)+  pow(marker.m_corners[0].y-marker.m_corners[1].y,2)   );
		dis[1]=sqrt( pow(marker.m_corners[1].x-marker.m_corners[2].x,2)+  pow(marker.m_corners[1].y-marker.m_corners[2].y,2)   );
		dis[2]=sqrt( pow(marker.m_corners[2].x-marker.m_corners[3].x,2)+  pow(marker.m_corners[2].y-marker.m_corners[3].y,2)   );
		dis[3]=sqrt( pow(marker.m_corners[3].x-marker.m_corners[0].x,2)+  pow(marker.m_corners[3].y-marker.m_corners[0].y,2)   );
		int min_dis=dis[0];
		for (int j = 0; j < 4; j++)
		        if(dis[j]<min_dis)
			  min_dis=dis[j];
	       // cout<<"min_side:"<<min_dis<<endl;
		if (min_dis < min_side_length)
			continue;
		  Rect rect1=boundingRect(contours[i]);
		    int width=rect1.width;
		    int height=rect1.height;
	        if (height*width >0.8*640*480)
			continue;
		drawContours(drawing, contours, i, color, 3, 8, vector<Vec4i>(), 0, Point());  
		
		Point2f v1 = marker.m_corners[1] - marker.m_corners[0];
		Point2f v2 = marker.m_corners[2] - marker.m_corners[0];
		if (v1.cross(v2) > 0)	//ÓÉÓÚÍŒÏñ×ø±êµÄYÖáÏòÏÂ£¬ËùÒÔŽóÓÚÁã²ÅŽú±íÄæÊ±Õë
		{
			swap(marker.m_corners[1], marker.m_corners[3]);
		}
		possible_markers.push_back(marker);
	}
	
       // imshow("Hull demo", drawing);  
//	cout<<"posiibale:"<<possible_markers.size()<<endl;
}

char  MarkerRecognizer:: check_point_front_arrow(float x,float y,float cx,float cy,float yaw)
{ 	
        float tyaw=90-yaw+0.000011;
	float kc_90=-1/tan(tyaw*0.0173);
	float bc_90=cy-kc_90*cx;
	float cx_t=cx+sin(yaw*0.0173)*1,cy_t=cy+cos(yaw*0.0173)*1;
	float flag[2];
	flag[0]=kc_90*cx_t+bc_90-cy_t;
	flag[1]=kc_90*x+bc_90-y;
	if((flag[0]>0&&flag[1]>0)||(flag[0]<0&&flag[1]<0))
	return 1;
	else 
  return 0;
}	
///-------------2D decoder
static int iLowH2d = 0, iHighH2d = 179, iLowS2d = 0, iHighS2d = 45, iLowV2d = 111, iHighV2d= 255;
void MarkerRecognizer::markerRecognize_2D(cv::Mat& img_rgb, vector<Marker1>& possible_markers, vector<Marker1>& final_markers)
{
       static int  init;
       static int g_nContraValue =55;
       static int g_nBrightValue = 50;
	static int TH = 125;//125;
	static int TL = 255;
       if(!init&&0){
	  namedWindow("c1", CV_WINDOW_AUTOSIZE); 
	  cvCreateTrackbar("C","c1",&g_nContraValue,255);
          cvCreateTrackbar("B","c1",&g_nBrightValue,255);
	  cvCreateTrackbar("TH","c1",&TH,255);
          cvCreateTrackbar("TL","c1",&TL,255);
	  cvCreateTrackbar("LowH", "c1", &iLowH2d, 179); //Hue (0 - 179)
	  cvCreateTrackbar("HighH", "c1", &iHighH2d, 179);

	  cvCreateTrackbar("LowS", "c1", &iLowS2d, 255); //Saturation (0 - 255)
	  cvCreateTrackbar("HighS", "c1", &iHighS2d, 255);

	  cvCreateTrackbar("LowV", "c1", &iLowV2d, 255); //Value (0 - 255)
	  cvCreateTrackbar("HighV", "c1", &iHighV2d, 255);
	 init=1;
      }
  
	final_markers.clear();
        bool good_marker =false; 
	Mat marker_image,img_gray;
	Mat bit_matrix(5, 5, CV_8UC1);
	Mat imgHSV,image,imgThresholded;
	vector<Mat> hsvSplit;
	img_rgb.copyTo(imgThresholded);
	/*cvtColor(image, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
	split(imgHSV, hsvSplit);
	equalizeHist(hsvSplit[2],hsvSplit[2]);
	merge(hsvSplit,imgHSV);
	Mat imgThresholded;
	inRange(imgHSV, Scalar(iLowH2d, iLowS2d, iLowV2d), Scalar(iHighH2d, iHighS2d, iHighV2d), imgThresholded); //Threshold the image
        Mat img_gray1,r1;*/
	//filteredRed(img_rgb,img_gray1,r1);
	//AdaptiveThereshold(img_rgb, imgThresholded,10 );
	cvtColor(imgThresholded, imgThresholded, CV_BGRA2GRAY);
	
	//imshow("ts",imgThresholded);
	
	for (int i = 0; i < possible_markers.size(); ++i)
	{
	  
	        Rect temp= boundingRect(possible_markers[i].m_corners);
		int iLowV2dt=limit(  20+ 100*( limit( temp.width*temp.height,0,20000)/20000) ,0,120);
		//inRange(imgHSV, Scalar(iLowH2d, iLowS2d, iLowV2dt), Scalar(iHighH2d, iHighS2d, iHighV2d), imgThresholded); //Threshold the image
                //cout<<"iLowV2dt="<<iLowV2dt<<"  S:"<<temp.width*temp.height<<endl;
		Mat img_gray1,r1;
		
		Mat M = getPerspectiveTransform(possible_markers[i].m_corners, m_marker_coords);
		warpPerspective(imgThresholded, marker_image, M, Size(MARKER_SIZE, MARKER_SIZE));
		threshold(marker_image, marker_image, TH, TL, THRESH_BINARY|THRESH_OTSU); //OTSU determins threshold automatically.
		//int thresh_size = (20/4)*2 + 1;
	      // adaptiveThreshold(marker_image, marker_image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 3, thresh_size/3);
		 //threshold(marker_image, marker_image, TH, TL, THRESH_BINARY); //OTSU determins threshold automatically.
	        Mat ele = getStructuringElement(MORPH_RECT, Size(5,5));
	        //morphologyEx(marker_image, marker_image, CV_MOP_BLACKHAT, ele);
		
		//imshow("2D coder origin",marker_image); //<<------------------------------------
		//A marker must has a whole black border.
		for (int y = 0; y < 7; ++y)
		{
			int inc = (y == 0 || y == 6) ? 1 : 6;
			int cell_y = y*MARKER_CELL_SIZE;
			for (int x = 0; x < 7; x += inc)
			{
				int cell_x = x*MARKER_CELL_SIZE;
				int none_zero_count = countNonZero(marker_image(Rect(cell_x, cell_y, MARKER_CELL_SIZE, MARKER_CELL_SIZE)));
				if (none_zero_count > MARKER_CELL_SIZE*MARKER_CELL_SIZE/4){
				         //cout<<"e1 :"<<none_zero_count<<"  "<< MARKER_CELL_SIZE*MARKER_CELL_SIZE/4<<endl;	
					goto __wrongMarker;
				      
				}
			}
		}
		//cout<<"e2"<<endl;	
		//Decode the marker
		for (int y = 0; y < 5; ++y)
		{
			int cell_y = (y+1)*MARKER_CELL_SIZE;

			for (int x = 0; x < 5; ++x)
			{
				int cell_x = (x+1)*MARKER_CELL_SIZE;
				int none_zero_count = countNonZero(marker_image(Rect(cell_x, cell_y, MARKER_CELL_SIZE, MARKER_CELL_SIZE)));
				if (none_zero_count > MARKER_CELL_SIZE*MARKER_CELL_SIZE/2)
					bit_matrix.at<uchar>(y, x) = 1;
				else
					bit_matrix.at<uchar>(y, x) = 0;
			}
		}

		//Find the right marker orientation
		//good_marker = true;
		int rotation_idx;	//ÄæÊ±ÕëÐý×ªµÄŽÎÊý
		for (rotation_idx = 0; rotation_idx < 4; ++rotation_idx)
		{
			if (hammingDistance(bit_matrix) == 0)
			{    
				good_marker = true;
				break;
			}
			bit_matrix = bitMatrixRotate(bit_matrix);
		}
		
		if (good_marker){
		  
		//Store the final marker
		Marker1& final_marker = possible_markers[i];
		
		final_marker.m_id = bitMatrixToId(bit_matrix);
		  if(bitMatrixToId(bit_matrix)>0){
		    std::rotate(final_marker.m_corners.begin(), final_marker.m_corners.begin() + rotation_idx, final_marker.m_corners.end());
		    final_markers.push_back(final_marker);
		  }
		}
	      __wrongMarker:
		continue;
	}
}

void MarkerRecognizer::markerRefine(cv::Mat& img_gray, vector<Marker1>& final_markers)
{
	for (int i = 0; i < final_markers.size(); ++i)
	{
		vector<Point2f>& corners = final_markers[i].m_corners;
		cornerSubPix(img_gray, corners, Size(5,5), Size(-1,-1), TermCriteria(CV_TERMCRIT_ITER, 30, 0.1));
	}
}

Mat MarkerRecognizer::bitMatrixRotate(cv::Mat& bit_matrix)
{
	//Rotate the bitMatrix by anti-clockwise way
	Mat out = bit_matrix.clone();
	int rows = bit_matrix.rows;
	int cols = bit_matrix.cols;

	for (int i=0; i<rows; ++i)
	{
		for (int j=0; j<cols; j++)
		{
			out.at<uchar>(i,j) = bit_matrix.at<uchar>(cols-j-1, i);
		}
	}
	return out;
}

int MarkerRecognizer::hammingDistance(Mat& bit_matrix)
{
	const int ids[4][5]=
	{
		{1,0,0,0,0},	// 00
		{1,0,1,1,1},	// 01
		{0,1,0,0,1},	// 10
		{0,1,1,1,0}		// 11
	};
  
	int dist=0;

	for (int y=0; y<5; ++y)
	{
		int minSum = INT_MAX; //hamming distance to each possible word
    
		for (int p=0; p<4; ++p)
		{
			int sum=0;
			//now, count
			for (int x=0; x<5; ++x)
			{
				sum += !(bit_matrix.at<uchar>(y, x) == ids[p][x]);
			}
			minSum = min(minSum, sum);
		}
    
		//do the and
		dist += minSum;
	}
  
	return dist;
}

int MarkerRecognizer::bitMatrixToId(Mat& bit_matrix)
{
	int id = 0;
	for (int y=0; y<5; ++y)
	{
		id <<= 1;
		id |= bit_matrix.at<uchar>(y,1);

		id <<= 1;
		id |= bit_matrix.at<uchar>(y,3);
	}
	return id;
}

vector<Marker1>& MarkerRecognizer::getMarkers()
{
	return m_markers;
}

vector<Marker1>& MarkerRecognizer::getMarkersc()
{
	return m_markers_c;
}

void MarkerRecognizer::drawToImage(cv::Mat& image, cv::Scalar color, float thickness)
{
	for (int i = 0; i < m_markers.size(); ++i)
	{
		m_markers[i].drawToImage(image, color, thickness,pos_2d,att_2d);	
	}
	#if USE_ARUCO
	for (unsigned int i = 0; i < Markersa.size(); i++)
        {	if(Markersa[i].id==1)
                 Markersa[i].draw(image, Scalar(0, 0, 255), 2);
        }
	Point text_point ;
	text_point.x = 20;
	text_point.y = 20;

	stringstream ss;
	ss <<"2D:"<<pos_2d;

	putText(image, ss.str(), text_point, FONT_HERSHEY_SIMPLEX,0.5, Scalar(0,0,250),2);
        text_point.x = 22;
	text_point.y = 240-44;
	stringstream s1s;
	s1s <<"2D_att"<<att_2d;
	putText(image, s1s.str(), text_point, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255),2);
	#endif
	
	if(m_markers_circle.size()){
	 ellipse(image, m_markers_circle[0], Scalar(0,255,255), 4, CV_AA);  
	 //ellipse(image, m_markers_circle[1], Scalar(0,0,255),2, CV_AA);  
	 circle(image,m_circle_head,8,Scalar(255),8);
	 circle(image,m_circle_center,1,Scalar(255),2);	  
	 Point text_point =m_circle_center;
	text_point.x = 22;
	text_point.y = 44;
	stringstream ss;
	ss <<"Circle"<<pos_circle;
	putText(image, ss.str(), text_point, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255),2);
        text_point.x = 22;
	text_point.y = 240-22;
	stringstream s1s;
	s1s <<"Circle_att"<<att_circle;
	putText(image, s1s.str(), text_point, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255),2);

	}
}

void MarkerRecognizer::img_mask(cv::Mat img_rgb,cv::Mat &out,cv::Mat mask)
{
  int i,j;
  int W=mask.rows;
  int H=mask.cols;
  for(i=0;i<W;i++)
     for(j=0;j<H;j++)
     {
        
                int cPointB=mask.at<Vec3b>(i,j)[0];  
	        int cPointG=mask.at<Vec3b>(i,j)[1];  
	        int cPointR=mask.at<Vec3b>(i,j)[2];  
		int dist=abs(cPointB - 255)/3 + abs(cPointG -255)/3 + abs(cPointR - 255)/3;
		if(dist<20){
		  out.at<Vec3b>(i,j)[0]=img_rgb.at<Vec3b>(i,j)[0];  
		  out.at<Vec3b>(i,j)[1]=img_rgb.at<Vec3b>(i,j)[1];  
		  out.at<Vec3b>(i,j)[2]=img_rgb.at<Vec3b>(i,j)[2];  
		}else{
		   out.at<Vec3b>(i,j)[0]=255;  
		  out.at<Vec3b>(i,j)[1]=255; 
		  out.at<Vec3b>(i,j)[2]=255;
		}
	          
    }
}


Point3d MarkerRecognizer::get_color(cv::Mat& img_rgb,Point center,int W)
{
  int i,j;
  int x=center.y,y=center.x;
  Point3d color;
  color.x=color.y=color.z=0;
  int cnt;
  for(i=x-W/2;i<x+W/2;i++)
     for(j=y-W/2;j<y+W/2;j++)
     {
        
                int cPointB=img_rgb.at<Vec3b>(i,j)[0];  
	        int cPointG=img_rgb.at<Vec3b>(i,j)[1];  
	        int cPointR=img_rgb.at<Vec3b>(i,j)[2];  
		// cout<<i<<" "<<j<<" "<<cPointB<<" "<<cPointG<<" "<<cPointR<<" "<<endl;
	  color.x+=cPointB;
	  color.y+=cPointG;
	  color.z+=cPointR;
	  cnt++;
    }
    color.x/=(cnt);
    color.y/=(cnt);
    color.z/=(cnt);
    
  return color;
}


float MarkerRecognizer::limit(float in,float min,float max)
{
  if(in>max)
    return max;
  else if(in<min)
    return min;
  else
    return in;
}


int MarkerRecognizer::check_bw(cv::Mat& image, std::vector<cv::Point2f> in_conner,cv::Scalar color,float thr,int dis)
{
  int cnt=0,i,j;
  int bx=in_conner[0].x;
  int by=in_conner[0].y;
  int mx=in_conner[0].x;
  int my=in_conner[0].y;
     for (i=0;i<4;i++)
      if(bx>in_conner[i].x)
	   bx=in_conner[i].x;
     for (i=0;i<4;i++)
      if(by>in_conner[i].y)
	   by=in_conner[i].y; 
      for (i=0;i<4;i++)
      if(mx<in_conner[i].x)
	   mx=in_conner[i].x; 
      for (i=0;i<4;i++)
      if(my<in_conner[i].y)
	   my=in_conner[i].y;   
      
     int size=abs(my-by)*abs(bx-mx);
     //cout<<"si:"<<size<<endl;
  for (i=bx;i<mx;i++)
      for (j=by;j<my;j++){
		int cPointB=image.at<Vec3b>(j,i)[0];  
	        int cPointG=image.at<Vec3b>(j,i)[1];  
	        int cPointR=image.at<Vec3b>(j,i)[2];  
		int dist=abs(cPointB - color[0])/3 + abs(cPointG - color[1])/3 + abs(cPointR - color[2])/3;
		if(dist<dis)
		   cnt++;
      }
 //  cout<<"cnt:"<<cnt<<endl;   
   if(cnt>thr*size)
     return 1;
   else 
     return 0;
}
