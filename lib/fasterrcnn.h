#ifndef FASTERRCNN_H
#define FASTERRCNN_H

#include <string>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>

#define FASTERRCNNSHARED_EXPORT

#define INPUT_SIZE_NARROW  600
#define INPUT_SIZE_LONG  1000

class FASTERRCNNSHARED_EXPORT FasterRcnn
{

public:

    /**
     * @brief FasterRcnn, default use gpu, if no gpu, #define CPU_ONLY
     * @param modelsPath
     */
    FasterRcnn(const std::string modelsPath);

    /**
     * @brief FasterRcnn
     * @param modelFile
     * @param weightFile
     */
    FasterRcnn(const std::string modelFile, const std::string weightFile);

    /**
     * @brief detect
     * @param img
     * @param rects
     */
    void detect(const cv::Mat& img, std::map<int, std::vector<cv::Rect> >& rects);

    /**
     * @brief detect
     * @param img
     * @param rects
     * @param confidences
     */
    void detect(const cv::Mat& img,
                std::map<int, std::vector<cv::Rect> >& objRects,
                std::map<int,std::vector<float> >& confidences);

private:

    void preProcess(const cv::Mat& img, cv::Mat& processedImg);

    void wrapInputLayer(std::vector<cv::Mat>* input_channels );


    boost::shared_ptr< caffe::Net<float> > net_;

    float m_fConfThresh;  //置信度阈值
    float m_fNmsThresh;   //非极大值抑制阈值
    float m_fImgScale;        //图像缩放尺度
};

#endif // FASTERRCNN_H
