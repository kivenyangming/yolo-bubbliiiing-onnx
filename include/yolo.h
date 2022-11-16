#pragma once
#include<iostream>
#include<math.h>
#include<opencv2/opencv.hpp>
struct Output {
	int id;//结果类别id
	float confidence;//结果置信度
	cv::Rect box;//矩形框
};

class Yolo
{
public:
	Yolo() {}
	~Yolo(){}
	bool readModel(cv::dnn::Net& net, std::string& netPath, bool isCuda);
	bool Detect(cv::Mat& SrcImg, cv::dnn::Net& net, std::vector<Output>& output);
	void drawPred(cv::Mat& img, std::vector<Output> result, std::vector<cv::Scalar> color);

private:
	//计算归一化函数
	float Sigmoid(float x) {
		return static_cast<float>(1.f / (1.f + exp(-x)));
	}

	//anchors
	const float netAnchors[3][6] = { { 12.0, 16.0,  19.0, 36.0,  40.0, 28.0 },{ 36.0, 75.0,  76.0, 55.0,  72.0, 146.0 },{ 142.0, 110.0,  192.0, 243.0,  459.0, 401.0 } };
	//stride
	const float netStride[3] = { 8.0, 16.0, 32.0 };
	const int netWidth = 640; //网络模型输入大小
	const int netHeight = 640;
	float nmsThreshold = 0.6;
	float boxThreshold = 0.6;
	float classThreshold = 0.6;
	//我的数据集类名
	std::vector<std::string> className = { "01", "02", "03", "04", "05", "06", "07","08",
                                           "11", "12", "13", "14", "15", "16", "17","18",
                                           "21", "22", "23", "24", "25", "26", "27","28",
                                           "31", "32", "33", "34", "35", "36", "37","38",
                                           "41", "42", "43", "44", "45", "46", "47","48",
                                           "51", "52", "53", "54", "55", "56", "57","58",
                                           "61", "62", "63", "64", "65", "66", "67","68",
                                           "71", "72", "73", "74", "75", "76", "77","78",
                                           "81", "82", "83", "84", "85", "86", "87","88",
                                           "91", "92", "93", "94", "95", "96", "97","98"};
};
