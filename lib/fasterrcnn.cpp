#include "fasterrcnn.h"

using namespace caffe;
using namespace cv;
using namespace std;

FasterRcnn::FasterRcnn(const std::string modelsPath)
{

}

FasterRcnn::FasterRcnn(const std::string modelFile, const std::string weightFile):
    m_fConfThresh(0.8),
    m_fNmsThresh(0.3)
{


#ifndef CPU_ONLY
    Caffe::set_mode(Caffe::GPU);
#else
    Caffe::set_mode(Caffe::CPU);
#endif

    net_.reset(new Net<float>(modelFile, TEST));
    net_->CopyTrainedLayersFrom(weightFile);

}

void FasterRcnn::detect(const cv::Mat& img, std::map<int, std::vector<cv::Rect> >& rects)
{

}


void FasterRcnn::detect(const cv::Mat& img,
                        std::map<int, std::vector<cv::Rect> >& objRects,
                        std::map<int,std::vector<float> >& confidences)
{
    //resize img and normalize
    Mat normalizedImg;
    preProcess(img, normalizedImg);

    int height = normalizedImg.rows;
    int width = normalizedImg.cols;

    caffe::shared_ptr<Blob<float> > input_layer = net_->blob_by_name("data");
    input_layer->Reshape(1, normalizedImg.channels(), height, width);
    net_->Reshape();

    //wrap input layer
    vector<cv::Mat> input_channels;
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i )
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += height * width;
    }

    // set data to net and do forward
    cv::split(normalizedImg, input_channels);

    float im_info[3];
    im_info[0] = height;
    im_info[1] = width;
    im_info[2] = m_fImgScale;

    net_->blob_by_name("im_info")->set_cpu_data(im_info);
    net_->Forward();


    // output of net
    int num = net_->blob_by_name("rois")->num();    //产生的 ROI 个数,比如为 13949个ROI
    const float *rois_data = net_->blob_by_name("rois")->cpu_data();    //维度比如为：13949*5*1*1
    cv::Mat rois_box(num, 4, CV_32FC1);
    for (int i = 0; i < num; ++i)
    {
        rois_box.at<float>(i, 0) = rois_data[i * 5 + 1] / m_fImgScale;
        rois_box.at<float>(i, 1) = rois_data[i * 5 + 2] / m_fImgScale;
        rois_box.at<float>(i, 2) = rois_data[i * 5 + 3] / m_fImgScale;
        rois_box.at<float>(i, 3) = rois_data[i * 5 + 4] / m_fImgScale;
    }

    caffe::shared_ptr<Blob<float> > bbox_delt_data = net_->blob_by_name("bbox_pred");   // 13949*84
    caffe::shared_ptr<Blob<float> > score = net_->blob_by_name("cls_prob");             // 3949*21


    int objCnt = net_->blob_by_name("cls_prob")->channels(); //目标类别数
    for (int i = 1; i < objCnt; ++i)  //对每个类，进行遍历
    {
        cv::Mat bbox_delt(num, 4, CV_32FC1);
        for (int j = 0; j < num; ++j){
            bbox_delt.at<float>(j, 0) = bbox_delt_data->data_at(j, i * 4 + 0, 0, 0);
            bbox_delt.at<float>(j, 1) = bbox_delt_data->data_at(j, i * 4 + 1, 0, 0);
            bbox_delt.at<float>(j, 2) = bbox_delt_data->data_at(j, i * 4 + 2, 0, 0);
            bbox_delt.at<float>(j, 3) = bbox_delt_data->data_at(j, i * 4 + 3, 0, 0);
        }
        cv::Mat box_class = RPN::bbox_tranform_inv(rois_box, bbox_delt);

        vector<RPN::abox> aboxes;   //对于 类别i，检测出的矩形框保存在这
        for (int j = 0; j < box_class.rows; ++j){
            if (box_class.at<float>(j, 0) < 0)  box_class.at<float>(j, 0) = 0;
            if (box_class.at<float>(j, 0) > (img.cols - 1))   box_class.at<float>(j, 0) = img.cols - 1;
            if (box_class.at<float>(j, 2) < 0)  box_class.at<float>(j, 2) = 0;
            if (box_class.at<float>(j, 2) > (img.cols - 1))   box_class.at<float>(j, 2) = img.cols - 1;

            if (box_class.at<float>(j, 1) < 0)  box_class.at<float>(j, 1) = 0;
            if (box_class.at<float>(j, 1) > (img.rows - 1))   box_class.at<float>(j, 1) = img.rows - 1;
            if (box_class.at<float>(j, 3) < 0)  box_class.at<float>(j, 3) = 0;
            if (box_class.at<float>(j, 3) > (img.rows - 1))   box_class.at<float>(j, 3) = img.rows - 1;
            RPN::abox tmp;
            tmp.x1 = box_class.at<float>(j, 0);
            tmp.y1 = box_class.at<float>(j, 1);
            tmp.x2 = box_class.at<float>(j, 2);
            tmp.y2 = box_class.at<float>(j, 3);
            tmp.score = score->data_at(j, i, 0, 0);
            aboxes.push_back(tmp);
        }


        // do nms and confidence selection
        std::sort(aboxes.rbegin(), aboxes.rend());
        RPN::nms(aboxes, m_fNmsThresh);
        for (int k = 0; k < aboxes.size();){
            if (aboxes[k].score < m_fConfThresh)
                aboxes.erase(aboxes.begin() + k);
            else
                k++;
        }

        // 将类别i的所有检测框，保存
        vector<cv::Rect> rect(aboxes.size());    //对于类别i，检测出的矩形框
        for(int ii=0;ii<aboxes.size();++ii)
            rect[ii] = cv::Rect(cv::Point(aboxes[ii].x1,aboxes[ii].y1),cv::Point(aboxes[ii].x2,aboxes[ii].y2));
        objRects[i] = rect;


        // 将类别i的所有检测框的打分，保存
        vector<float> tmp(aboxes.size());       //对于 类别i，检测出的矩形框的得分
        for(int ii=0;ii<aboxes.size();++ii)
            tmp[ii]=aboxes[ii].score;

        confidences.insert(pair<int,vector<float> >(i,tmp));

    }

}


void FasterRcnn::preProcess(const Mat &img, Mat& processedImg)
{

    //计算图像缩放尺度
    int max_side = max(img.rows, img.cols);                             //分别求出图片宽和高的较大者
    int min_side = min(img.rows, img.cols);
    float max_side_scale = float(max_side) / float(INPUT_SIZE_LONG);    //分别求出缩放因子
    float min_side_scale = float(min_side) / float(INPUT_SIZE_NARROW);
    float max_scale = max(max_side_scale, min_side_scale);

    m_fImgScale = float(1) / max_scale;

    int height = int(img.rows * m_fImgScale);
    int width = int(img.cols * m_fImgScale);

    //normalize img

    Mat cv_resized;
    resize( img, cv_resized, cv::Size(width, height) );
    cv_resized.convertTo( cv_resized, CV_32FC3 );

    Mat mean( height, width, cv_resized.type(), cv::Scalar(102.9801, 115.9465, 122.7717) );
    subtract( cv_resized, mean, processedImg );
}
