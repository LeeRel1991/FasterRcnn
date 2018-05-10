#include <QCoreApplication>
#include "../lib/fasterrcnn.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;


string num2str(float i){
    stringstream ss;
    ss<<i;
    return ss.str();
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    if(argc<7)
    {
        cout<<"usage: \n "
              "./samples --modelFile $prototxt_model_file --weightFile $trained_weight_file --video $video_file_to_detect"<<endl;
        cout<<"eg: ./samples --modelFile ZF-test.prototxt --weightFile ZF_faster_rcnn_final.caffemodel --video /media/lirui/Program/Datas/Videos/face.mp4"<<endl;
        return -1;
    }

    std::string modelFile, weightFile, videoFile;
    for (int i =0; i <argc; ++i)
    {
        if (std::string(argv[i]) == "--modelFile"){
            modelFile = argv[i+1];
        }
        if (std::string(argv[i]) == "--weightFile"){
            weightFile = argv[i+1];
        }
        if (std::string(argv[i]) == "--video"){
            videoFile = argv[i+1];
        }

    }

    FasterRcnn detector(modelFile, weightFile);

    cv::VideoCapture capture;
    capture.open(videoFile);
    if (!capture.isOpened())
    {
       std::cout << "视频读取失败！" << std::endl;
    }

    std::clock_t start, end;
    double total_time =0;

    Mat imgFrame;
    while(1)
    {

        capture >> imgFrame;
        map<int,vector<float> > score;
        start = std::clock();

        map<int,vector<Rect> > label_objs;
        detector.detect(imgFrame, label_objs, score);  //目标检测,同时保存每个框的置信度

        end = std::clock();
        total_time = (double)(end -start) / CLOCKS_PER_SEC;
        std::cout << "time: " << total_time << "s" <<std::endl;


        for(map<int,vector<Rect> >::iterator it=label_objs.begin();it!=label_objs.end();it++){
            int label=it->first;  //标签
            vector<Rect> rects=it->second;  //检测框
            for(int j=0;j<rects.size();j++) {
                rectangle(imgFrame,rects[j],Scalar(0,0,255),2);   //画出矩形框
                string txt=num2str(label)+" : "+num2str(score[label][j]);
                putText(imgFrame,txt,Point(rects[j].x,rects[j].y),CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0)); //标记 类别：置信度
            }
        }

        imshow("1", imgFrame);
        waitKey(1);


    }

    return a.exec();
}
