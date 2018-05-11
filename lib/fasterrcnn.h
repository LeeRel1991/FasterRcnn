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
     * @param modelsPath, directory of models path, consist of model file, weight file and labels file，
     * three files use the default name: "faster_rcnn_test.prototxt", "faster_rcnn_final.caffemodel", "classes.txt";
     */
    FasterRcnn(const std::string modelsPath);

    /**
     * @brief FasterRcnn
     * @param modelFile, prototxt defines tha cnn network
     * @param weightFile, params trained on dataset
     * @param labelsFile, label name for each id in cnn's output
     */
    FasterRcnn(const std::string modelFile, const std::string weightFile, const std::string labelsFile);


    ~FasterRcnn(){}

    /**
     * @brief detect, detect objects on @em img
     * @param img
     * @param rects, pairs of each kind of object and the detected pixel boundingbox position on @em img
     */
    void detect(const cv::Mat& img, std::map<std::string, std::vector<cv::Rect> >& rects);

    /**
     * @brief detect
     * @param img
     * @param objRects
     * @param confidences, the corresponding confidence score (0~1) of each boundingbox
     */
    void detect(const cv::Mat& img,
                std::map<std::string, std::vector<cv::Rect> >& objRects,
                std::map<std::string, std::vector<float> >& confidences);

private:

    void initNet(const std::string modelFile, const std::string weightFile, const std::string labelsFile);
    void preProcess(const cv::Mat& img, cv::Mat& processedImg);

    void wrapInputLayer(std::vector<cv::Mat>* input_channels );


    boost::shared_ptr< caffe::Net<float> >  net_;
    std::vector<std::string>                m_labels;
    float m_fConfThresh;        //置信度阈值
    float m_fNmsThresh;         //非极大值抑制阈值
    float m_fImgScale;          //图像缩放尺度
};

#endif // FASTERRCNN_H
