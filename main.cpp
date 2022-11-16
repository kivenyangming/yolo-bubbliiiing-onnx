#include "yolo.h"
#include <iostream>
#include<opencv2//opencv.hpp>
#include<math.h>
//#include<windows.h>
//#pragma comment( linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"" ) // 设置入口地址

using namespace std;
using namespace cv;
using namespace dnn;

int main()
{
	//string img_path;
	//getline(cin, img_path);
	string img_path = "./640.jpg";
	string model_path = "./yolov7b.onnx";

	Yolo test;
	Net net;
    //加载onnx模型
	if (test.readModel(net, model_path, true)) {
		cout << "read net ok!" << endl;
	}
	else {
		return -1;
	}

	//生成随机颜色
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) { //你的类别数量
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	vector<Output> result;


	Mat img = imread(img_path);
	if (test.Detect(img, net, result)) {
		test.drawPred(img, result, color);
	}
	else {
		cout << "Detect Failed!" << endl;
	}

	return 0;
}