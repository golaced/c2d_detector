  #ifndef __MARKER_RECOGNIZER_H__
#define __MARKER_RECOGNIZER_H__

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
// Aruco libraries
#define USE_ARUCO 1
#if USE_ARUCO
#include <aruco/aruco.h>
#include <aruco/cameraparameters.h>
#include <aruco/cvdrawingutils.h>
#endif
extern float att_drone[3];
class Marker1
{
public:
	int m_id;
	std::vector<cv::Point2f> m_corners;
	cv::Point2f c_corners[9];
	// c0------c3
	// |		|
	// |		|
	// c1------c2

public:
	Marker1();
	Marker1(int m);
	Marker1(int _id, cv::Point2f _c0, cv::Point2f _c1, cv::Point2f _c2, cv::Point2f _c3);

	void estimateTransformToCamera(std::vector<cv::Point3f> corners_3d, cv::Mat& camera_matrix, cv::Mat& dist_coeff, cv::Mat& rmat, cv::Mat& tvec);
	void drawToImage(cv::Mat& image, cv::Scalar color, float thickness, cv::Point3f pos_2d, cv::Point3f att_2d);
	//void drawToImage_single(cv::Mat& image, std::vector<cv::Point2f>in_conner,cv::Scalar color, float thickness);
};

class MarkerRecognizer
{
private:
	std::vector<cv::Point2f> m_marker_coords;
	cv::Point m_circle_center,m_circle_head;
	cv::Mat camera_matrix;//内参数矩阵
	cv::Mat distortion_coefficients;//畸变系数
	cv::Point2f coner_2D[4],coner_color[4],coner_circle[4];
	float k_circle,cr1,cr2,r2d;
	#if USE_ARUCO
	int aruco_size;
	#endif
private:
	void markerDetect(cv::Mat& img_rgb, std::vector<Marker1>& possible_markers, int min_size, int min_side_length);
	void markerDetect_circle(cv::Mat& img_rgb,cv::Mat& aruco_thr,  std::vector<Marker1>& possible_markers, int min_size, int min_side_length);
	void markerRecognize_2D(cv::Mat& img_rgb, std::vector<Marker1>& possible_markers, std::vector<Marker1>& final_markers);
	void markerRecognize_color(cv::Mat& img_rgb,std::vector<Marker1>& possible_markers, std::vector<Marker1>& final_markers);
	void markerRefine(cv::Mat& img_gray, std::vector<Marker1>& final_markers);
	void Color_tag_refine(cv::Mat& Img_rgb, std::vector<Marker1>& possible_markers,std::vector<Marker1>& final_markers);
	cv::Mat bitMatrixRotate(cv::Mat& bit_matrix);
	int hammingDistance(cv::Mat& bit_matrix);
	int bitMatrixToId(cv::Mat& bit_matrix);
	void transform1(double x_in, double y_in, double& x_out, double& y_out) ;
        void estimateTransformToCamera_all(cv::Mat  cimage,cv::Mat& rmat, cv::Mat& tvec);	
	void getCameraPos(cv::Mat Rvec, cv::Mat Tvec, cv::Point3f &pos,cv::Point3f &att);
	#if USE_ARUCO
	void getCameraPosa(cv::Mat Rvec, cv::Mat Tvec, cv::Point3f &pos);
	void getAttitudea(aruco::Marker marker, cv::Point3f &attitude);
	#endif
	float To_180_degrees(float x);
	cv::Point3d get_color(cv::Mat& img_rgb,cv::Point center,int W);
	float limit(float in,float min,float max);
	int check_bw(cv::Mat& image, std::vector<cv::Point2f> in_conner,cv::Scalar color,float thr,int dis);
	void RodriguesR(cv::Matx13d normal_mat,cv::Vec3f &rot);
	void bubbleSort(float* pData, int * id,int length);
	void ProjectCheck(cv::Vec3f rot,cv::Vec3f pos,  cv::Point3f pos3d ,cv:: Point2f uv,cv:: Point2f &uv_p,  cv::Point3f & err);
	void img_mask(cv::Mat img_rgb,cv::Mat& out,cv::Mat mask);
	int  color_detect1(cv::Mat& in,int* lines);
	char  check_point_front_arrow(float x,float y,float cx,float cy,float yaw);
public:
	std::vector<Marker1> m_markers;
	cv::Point2f center_circle,center_2d;
	#if USE_ARUCO
	std::vector< aruco::Marker > Markersa;
	#endif
        std::vector<Marker1> m_markers_c;
        std::vector <cv::RotatedRect> m_markers_circle;
	MarkerRecognizer();
	 void init_marker(float c_2d,float c_color, float r1, float r2 );
	int update(cv::Mat& in, int min_size, int min_side_length = 10);
	std::vector<Marker1>& getMarkers();
	std::vector<Marker1>& getMarkersc();
        cv::Point3f  pos_2d,pos_color,pos_circle;
	cv::Point3f  att_2d,att_color,att_circle;
	void drawToImage(cv::Mat& image, cv::Scalar color, float thickness);
	
		//设置相机内参数矩阵
	void SetCameraMatrix(double fx, double fy, double u0, double v0)
	{
		camera_matrix = cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0));
		camera_matrix.ptr<double>(0)[0] = fx;
		camera_matrix.ptr<double>(0)[2] = u0;
		camera_matrix.ptr<double>(1)[1] = fy;
		camera_matrix.ptr<double>(1)[2] = v0;
		camera_matrix.ptr<double>(2)[2] = 1.0f;
	}
	//设置畸变系数矩阵
	void SetDistortionCoefficients(double k_1, double  k_2, double  p_1, double  p_2, double k_3)
	{
		distortion_coefficients = cv::Mat(5, 1, CV_64FC1, cv::Scalar::all(0));
		distortion_coefficients.ptr<double>(0)[0] = k_1;
		distortion_coefficients.ptr<double>(1)[0] = k_2;
		distortion_coefficients.ptr<double>(2)[0] = p_1;
		distortion_coefficients.ptr<double>(3)[0] = p_2;
		distortion_coefficients.ptr<double>(4)[0] = k_3;
	}
	
	//void drawToImage_single(cv::Mat& image, std::vector<cv::Point2f>in_conner,cv::Scalar color, float thickness);
};

#endif
