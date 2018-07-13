#include <ros/ros.h>
#include<image_transport/image_transport.h>
#include<cv_bridge/cv_bridge.h>
#include<sensor_msgs/image_encodings.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <nav_msgs/Odometry.h> 
#include <std_msgs/ByteMultiArray.h>
#include "mark_deteck.h"
MarkerRecognizer m_recognizer;
static const std::string OPENCV_WINDOW = "Image window";
using namespace cv;
using namespace std;
cv::Mat image,image_rgb,show;
void imageCb(const sensor_msgs::ImageConstPtr& msg)
{
cv_bridge::CvImagePtr cv_ptr;
try
{
cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
}
catch (cv_bridge::Exception& e)
{
ROS_ERROR("cv_bridge exception: %s", e.what());
return;
}

cv::Size InImage_size(640,480);
image=cv_ptr->image;
resize(image, image, InImage_size);
image.copyTo(image_rgb) ;
image.copyTo(show) ;
}


float To_180_degrees(float x)
{
	return (x>180?(x-360):(x<-180?(x+360):x));
}

void q_to_eular(float x,float y,float z,float w,float att[3],float off)
{

att[0] = atan2(2 * (y*z + w*x), w*w - x*x - y*y + z*z)*57.3;
att[1] = asin(-2 * (x*z - w*y))*57.3;
att[2] = atan2(2 * (x*y + w*z), w*w + x*x - y*y - z*z)*57.3;
att[2] = To_180_degrees(atan2(2 * (-x*y - w*z), 2*(w*w+x*x)-1)*57.3+off);

}


nav_msgs::Odometry drone;
float att_drone[3];
void drone_cb(const nav_msgs::Odometry &msg)
{
static int show1;
    //ROS_INFO("Received a drone odom message!");  
    //ROS_INFO("Drone Position:[%f,%f,%f]",msg.pose.pose.position.x ,msg.pose.pose.position.y,msg.pose.pose.position.z);  
    //ROS_INFO("Spd Components:[%f,%f,%f]",msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.angular.z);  
	drone=msg;
	q_to_eular(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,
	msg.pose.pose.orientation.z,msg.pose.pose.orientation.w,att_drone,90);
 if(show1++>0){show1=0;
 // ROS_INFO("Drone Att:[%f,%f,%f]",att_drone[0],att_drone[1],att_drone[2]);  
  //ROS_INFO("Car Att:[%f,%f,%f]",car_drone[0],car_drone[1],car_drone[2]);  
  }
}

geometry_msgs::Vector3 attd;
void att_cb(const geometry_msgs::Vector3 &msg)
{
	attd=msg;
	att_drone[0]=attd.x;
	att_drone[1]=attd.y;
	att_drone[2]=attd.z;
}




std_msgs::ByteMultiArray data_to_publish;
cv::Point3f coordinate_camera(0, 0, 0);
cv::Point3f atti_camera;
cv::Point2f markerorigin_img(0,0);
void format_data_to_send(int marker_size)
{
    unsigned char data_to_send[50];
    int Length = 0;
	int _cnt = 0, i = 0, sum = 0;
    int Locate[5] = {0,0,0,0,0};

	data_to_send[_cnt++] = 0xAA;
	data_to_send[_cnt++] = 0xAF;
	data_to_send[_cnt++] = 0x21;
	data_to_send[_cnt++] = 0;

	data_to_send[_cnt++] = marker_size;
	data_to_send[_cnt++] = int(coordinate_camera.x*100) >> 8;
	data_to_send[_cnt++] = int(coordinate_camera.x*100) % 256;
	data_to_send[_cnt++] = int(coordinate_camera.y*100) >> 8;
	data_to_send[_cnt++] = int(coordinate_camera.y*100) % 256;
	data_to_send[_cnt++] = int(coordinate_camera.z*100) >> 8;
	data_to_send[_cnt++] = int(coordinate_camera.z*100) % 256;
	data_to_send[_cnt++] = int(atti_camera.x) >> 8;
	data_to_send[_cnt++] = int(atti_camera.x) % 256;
	data_to_send[_cnt++] = int(atti_camera.y) >> 8;
	data_to_send[_cnt++] = int(atti_camera.y) % 256;
	data_to_send[_cnt++] = int(atti_camera.z) >> 8;
	data_to_send[_cnt++] = int(atti_camera.z) % 256;


    if(marker_size == 0)
    {
        data_to_send[_cnt++] = 0;
        data_to_send[_cnt++] = 0;
        data_to_send[_cnt++] = 0;
        data_to_send[_cnt++] = 0;
	data_to_send[_cnt++] = 0;
        data_to_send[_cnt++] = 0;
    }
    else
    {
        data_to_send[_cnt++] = int(markerorigin_img.x) >> 8;
        data_to_send[_cnt++] = int(markerorigin_img.x) % 256;
        data_to_send[_cnt++] = int(markerorigin_img.y) >> 8;
        data_to_send[_cnt++] = int(markerorigin_img.y) % 256;
    }

    //for(size_t i=0;i<objectLocate.size();i++)
    //{
    //   Locate[objectLocate[i]] = bool(objectLocate[i]);
    //}

    data_to_send[_cnt++] = 0;//Locate[1];
    data_to_send[_cnt++] = 0;//Locate[2];
    data_to_send[_cnt++] = 0;//Locate[3];
    data_to_send[_cnt++] = 0;//Locate[4];
    
    data_to_send[_cnt++] = 0;//int(markerorigin_img.val[2]) >> 8;
    data_to_send[_cnt++] = 0;//int(markerorigin_img.val[2]) % 256;
    data_to_send[_cnt++] = 0;//int(markerorigin_img.val[3]) >> 8;
    data_to_send[_cnt++] = 0;//int(markerorigin_img.val[3]) % 256;

	data_to_send[3] = _cnt - 4;

	for (i = 0; i < _cnt; i++)
		sum += data_to_send[i];
	data_to_send[_cnt++] = sum;
	Length = _cnt;

	data_to_publish.data.clear();
	for (int i = 0; i < _cnt; i++)
	{
		data_to_publish.data.push_back(data_to_send[i]);
	}
}


#define CAP_WIDTH_320 1
int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ros::NodeHandle nh;
  ros::Subscriber sub_drone = nh.subscribe("/ground_truth/state", 1, drone_cb);
  ros::Subscriber sub_att = nh.subscribe("/att_drone", 1, att_cb);
  ros::Subscriber sub_camera= nh.subscribe("/ardrone/45/image_raw", 1,imageCb);
  ros::Publisher qrland_info_pub = nh.advertise<std_msgs::ByteMultiArray>("oldx/oldx_send", 1);
  ros::Publisher  pub_w2c = nh.advertise<nav_msgs::Odometry>("/w2c_pos", 1);
  // try opening first as video
  VideoCapture cap(0);
  if(!cap.isOpened())  
    {  
        return -1;  
    }  
#if CAP_WIDTH_320
cap.set(CV_CAP_PROP_FRAME_WIDTH , 320);
cap.set(CV_CAP_PROP_FRAME_HEIGHT , 240);
#else
cap.set(CV_CAP_PROP_FRAME_WIDTH , 640);
cap.set(CV_CAP_PROP_FRAME_HEIGHT , 480);
#endif
	double fx = 420  ;
	double fy = fx  ;
	double u0 = 202;
	double v0 = 99;
	//镜头畸变参数
	int en_k=0;
	double k1 = 0.20032167*en_k;
	double k2 = 0.43894774*en_k;
	double p1 = -0.0057549393*en_k;
	double p2 = 0.0043098251*en_k;
	double k3 = -5.4617381*en_k;
	float out_r=0.40;//0.75;
	float in_r=71./130.*out_r;
	float r_2d=42./130.*out_r;
	m_recognizer.init_marker(r_2d,r_2d*0.4,out_r/2,in_r/2);
	m_recognizer.SetCameraMatrix(fx, fy, u0, v0);
	m_recognizer.SetDistortionCoefficients(k1, k2, p1, p2, k3);

	ros::Rate loop_rate(500);
	//cv::namedWindow(OPENCV_WINDOW);
	while (ros::ok())
	{
	   Mat image_rgb1,show1;
	   cap >> image_rgb1;
	   cv::Size InImage_size(320,240);
	   resize(image_rgb1, image_rgb1, InImage_size);
	   image_rgb1.copyTo(show1) ;
	   if (image_rgb1.empty()) break;
		
		m_recognizer.update(image_rgb1, 120,20);
		m_recognizer.drawToImage(show1,Scalar(255,0,255),2);
		vector<Marker1>& markers = m_recognizer.getMarkers();

		// Update GUI Window
		cv::imshow("OPENCV_WINDOW",show1);
		
		int marker_size=0;
		nav_msgs::Odometry temp;
		//circle
		if(m_recognizer.m_markers_circle.size()){
		marker_size++;
		temp.pose.pose.position.x=m_recognizer.pos_circle.x;
		temp.pose.pose.position.y=m_recognizer.pos_circle.y;
		temp.pose.pose.position.z=m_recognizer.pos_circle.z;
		temp.twist.twist.angular.x=m_recognizer.att_circle.z;
		}
		//2d
		#if USE_ARUCO
		if(m_recognizer.Markersa.size()){
		#else
		if(m_recognizer.m_markers.size()){
		#endif
		marker_size++;
		temp.twist.twist.linear.x=m_recognizer.pos_2d.x;
		temp.twist.twist.linear.y=m_recognizer.pos_2d.y;
		temp.twist.twist.linear.z=m_recognizer.pos_2d.z;
		temp.twist.twist.angular.z=m_recognizer.att_2d.z;
		}
	        pub_w2c.publish(temp);
		markerorigin_img.x=markerorigin_img.y=0;
		coordinate_camera.x=coordinate_camera.y=coordinate_camera.z=atti_camera.z=0;
		float flt_2d=0.5;
		
		if(m_recognizer.m_markers_circle.size()){
		coordinate_camera.x=m_recognizer.pos_circle.x;
		coordinate_camera.y=m_recognizer.pos_circle.y;
		coordinate_camera.z=m_recognizer.pos_circle.z;
		atti_camera.z=m_recognizer.att_circle.z;
		markerorigin_img.x=m_recognizer.center_circle.x;
		markerorigin_img.y=m_recognizer.center_circle.y;
		}
		#if USE_ARUCO
		if(m_recognizer.Markersa.size()){
		#else
		if(m_recognizer.m_markers.size()){
		#endif
		coordinate_camera.x=m_recognizer.pos_2d.x;
		coordinate_camera.y=m_recognizer.pos_2d.y;
		coordinate_camera.z=m_recognizer.pos_2d.z;
		atti_camera.z=m_recognizer.att_2d.z;
		markerorigin_img.x=m_recognizer.center_2d.x;
		markerorigin_img.y=m_recognizer.center_2d.y;
		}
		
		#if USE_ARUCO
		if(m_recognizer.Markersa.size()&&m_recognizer.m_markers_circle.size()){
		#else
		if(m_recognizer.m_markers.size()&&m_recognizer.m_markers_circle.size()){
		#endif
		if(m_recognizer.pos_2d.z<0.5)
		 flt_2d=0.6;
		else
		 flt_2d=0.3;
		coordinate_camera.x=m_recognizer.pos_2d.x*flt_2d+(1-flt_2d)*m_recognizer.pos_circle.x;
		coordinate_camera.y=m_recognizer.pos_2d.y*flt_2d+(1-flt_2d)*m_recognizer.pos_circle.y;
		coordinate_camera.z=m_recognizer.pos_2d.z*flt_2d+(1-flt_2d)*m_recognizer.pos_circle.z;
		atti_camera.z=m_recognizer.att_2d.z*flt_2d+(1-flt_2d)*m_recognizer.att_circle.z;
		}
		static int cnt_show;
		if(cnt_show++>4){cnt_show=0;
		cout<<coordinate_camera<<" "<<atti_camera.z<<endl;
		}
		
		format_data_to_send(marker_size);
                qrland_info_pub.publish(data_to_publish);


		  char c_key = cv::waitKey(5);
		    if (c_key == 27) // wait for key to be pressed
		    {
			break;
		    }
		ros::spinOnce();
		loop_rate.sleep();
	}
  return 0;
}
