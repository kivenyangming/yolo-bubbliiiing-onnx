#include "Yolo.h"
using namespace std;
using namespace cv;
using namespace dnn;

bool Yolo::readModel(Net& net, string& netPath, bool isCuda = false) {
	try {
		net = readNetFromONNX(netPath);
	}
	catch (const std::exception&) {
		return false;
	}
	//cuda
	if (isCuda) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	//cpu
	else {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	return true;
}


bool Yolo::Detect(Mat& SrcImg, Net& net, vector<Output>& output) {
	Mat blob;
	int col = SrcImg.cols;
	int row = SrcImg.rows;
	int maxLen = MAX(col, row);
	Mat netInputImg = SrcImg.clone();
	if (maxLen > 1.2 * col || maxLen > 1.2 * row) {
		Mat resizeImg = Mat::zeros(maxLen, maxLen, CV_8UC3);
		SrcImg.copyTo(resizeImg(Rect(0, 0, col, row)));
		netInputImg = resizeImg;
	}
	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(115, 115, 115), true, false);



	net.setInput(blob);
	std::vector<cv::Mat> netOutputImg;
	net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
	
	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	float ratio_h = (float)netInputImg.rows / netHeight;
	float ratio_w = (float)netInputImg.cols / netWidth;
	int net_width = className.size() + 5;  //输出的网络宽度是类别数+5
	float* pdata = (float*)netOutputImg[0].data; //按照指针寻找
	for (int stride = 0; stride < 3; stride++) {    //stride
		int grid_x = (int)(netWidth / netStride[stride]);
		int grid_y = (int)(netHeight / netStride[stride]);
		for (int anchor = 0; anchor < 3; anchor++) { //anchors
			const float anchor_w = netAnchors[stride][anchor * 2];
			const float anchor_h = netAnchors[stride][anchor * 2 + 1];
			for (int i = 0; i < grid_x; i++) {
				for (int j = 0; j < grid_y; j++) {
					float box_score = Sigmoid(pdata[4]);//获取每一行的box框中含有某个物体的概率
					if (box_score > boxThreshold) {
						//为了使用minMaxLoc(),将85长度数组变成Mat对象
						cv::Mat scores(1, className.size(), CV_32FC1, pdata + 5);
						Point classIdPoint;
						double max_class_socre;
						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						max_class_socre = Sigmoid((float)max_class_socre);
						if (max_class_socre > classThreshold) {
							//rect [x,y,w,h]
							float x = (Sigmoid(pdata[0]) * 2.f - 0.5f + j) * netStride[stride];  //x
							float y = (Sigmoid(pdata[1]) * 2.f - 0.5f + i) * netStride[stride];   //y
							float w = powf(Sigmoid(pdata[2]) * 2.f, 2.f) * anchor_w;   //w
							float h = powf(Sigmoid(pdata[3]) * 2.f, 2.f) * anchor_h;  //h
							int left = (x - 0.5 * w) * ratio_w;
							int top =  (y - 0.5 * h) * ratio_h;
							classIds.push_back(classIdPoint.x);
							confidences.push_back(max_class_socre * box_score);
							boxes.push_back(Rect(left, top, int(w * ratio_w), int(h * ratio_h)));
						}
					}
					pdata += net_width;//指针移到下一行
				}
			}
		}
	}
	vector<int> nms_result;
	dnn::NMSBoxes(boxes, confidences, classThreshold, nmsThreshold, nms_result);
	for (int i = 0; i < nms_result.size(); i++) {
		int idx = nms_result[i];
		Output result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		output.push_back(result);
	}

	if (output.size())
		return true;
	else
		return false;
}


void Yolo::drawPred(Mat& img, vector<Output> result, vector<Scalar> color) {
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 8);

		string label = className[result[i].id] + ":" + to_string(result[i].confidence);

		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	imshow("out", img);
	waitKey();
}