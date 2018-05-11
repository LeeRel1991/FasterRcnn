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


    if (argc < 5)
    {
        cout<<"usage: \n "
          "./samples --modelFile $prototxt_model_file --weightFile $trained_weight_file --video $video_file_to_detect \n" << endl;
        cout<<" or: \n"
          "./samples --modelPath path to contains all the required files(model. weight, class) --video $video_file_to_detect \n" << endl;


        cout<<"eg: ./samples --modelFile ZF-test.prototxt --weightFile ZF_faster_rcnn_final.caffemodel --video /media/lirui/Program/Datas/Videos/face.mp4"<<endl;
        cout<<"or:\n"
              "./samples --modelPath ./objects-VGG16 --video /media/lirui/Program/Datas/Videos/face.mp4"<<endl;

        return -1;
    }

    FasterRcnn* detector;
    string videoFile;
    for (int i =0; i <argc; ++i)
    {
        if (std::string(argv[i]) == "--video"){
            videoFile = argv[i+1];
        }
    }

    if (argc < 9 )
    {
        std::string modelPath ;
        for (int i =0; i <argc; ++i)
        {

            if (std::string(argv[i]) == "--modelPath"){
                modelPath = argv[i+1];
            }
        }

        detector = new FasterRcnn(modelPath);

    }
    else
    {
        std::string modelFile, weightFile, labelsFile;
        for (int i =0; i <argc; ++i)
        {

            if (std::string(argv[i]) == "--modelFile"){
                modelFile = argv[i+1];
            }
            if (std::string(argv[i]) == "--weightFile"){
                weightFile = argv[i+1];
            }
            if (std::string(argv[i]) == "--labelsFile"){
                labelsFile = argv[i+1];
            }

            detector= new FasterRcnn(modelFile, weightFile, labelsFile);

        }
    }

    cv::VideoCapture capture;
    capture.open("/media/lirui/Program/Datas/Videos/face.mp4");
    if (!capture.isOpened())
    {
       std::cout << "视频读取失败！" << std::endl;
    }

    std::clock_t start, end;
    double total_time =0;

    Mat imgFrame;
    int ii=0;
    while(++ii<200)
    {

        capture >> imgFrame;
        map<string, vector<float> > score;
        start = std::clock();

        map<string, vector<Rect> > label_objs;
        detector->detect(imgFrame, label_objs, score);  //目标检测,同时保存每个框的置信度

        end = std::clock();
        total_time = (double)(end -start) / CLOCKS_PER_SEC;
        std::cout << "time: " << total_time << "s" <<std::endl;


        for(map<string, vector<Rect> >::iterator it=label_objs.begin();it!=label_objs.end();it++){
            string label = it->first;       //标签
            vector<Rect> rects=it->second;  //检测框
            for(int j=0; j<rects.size(); j++) {
                rectangle(imgFrame, rects[j],Scalar(0,0,255),2);   //画出矩形框
                putText(imgFrame, label, Point(rects[j].x,rects[j].y),CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0)); //标记 类别：置信度
            }
        }

        imshow("1", imgFrame);
        waitKey(1);

    }

    delete detector;

    return a.exec();
}
