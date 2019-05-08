#ifndef IREASONINGMACHINE_H
#define IREASONINGMACHINE_H
#include <opencv2/opencv.hpp>

class IReasoningMachine
{
public:
    IReasoningMachine(std::string  model_path, int nNetWorkWidth, int nNetWorkHeight):
             m_model_path(model_path),m_nNetWorkWidth(nNetWorkWidth),m_nNetWorkHeight(nNetWorkHeight){}
    virtual ~IReasoningMachine() {}

public:
    virtual bool Init() = 0;
    virtual bool UnInit() = 0;
//public:
    virtual bool  ReasonImage(cv::Mat &images) = 0;
    virtual void GetReasonResult(void* outputData, unsigned int outputDataLen) = 0;

protected:
    void preprocess_image(const cv::Mat& src_image_mat, cv::Mat& preprocessed_image_mat)
    {
        cv::Mat src_image_mat_temp;
        cv::resize(src_image_mat, src_image_mat_temp, cv::Size(m_nNetWorkWidth,m_nNetWorkHeight));
        cv::cvtColor(src_image_mat_temp, preprocessed_image_mat, CV_BGR2RGB);     //CV_BGR2RGB
        //whiten_image(preprocessed_image_mat, result_image_mat);
        //mean and std
        //c * r * 3 => c * 3r * 1
        cv::Mat temp = preprocessed_image_mat.reshape(1, preprocessed_image_mat.rows * 3);

        cv::Mat mean3;
        cv::Mat stddev3;
        cv::meanStdDev(temp, mean3, stddev3);
        double mean_pxl = mean3.at<double>(0);
        double stddev_pxl = stddev3.at<double>(0);

        //std::cout << mean_pxl << "  " << stddev_pxl << std::endl;
        preprocessed_image_mat.convertTo(preprocessed_image_mat, CV_64FC3);
        preprocessed_image_mat = preprocessed_image_mat - cv::Vec3d(mean_pxl, mean_pxl, mean_pxl);
        preprocessed_image_mat = preprocessed_image_mat / stddev_pxl;
    }
protected:

    std::string  m_model_path;
    int m_nNetWorkWidth;
    int m_nNetWorkHeight;
};

#endif // IREASONINGMACHINE_H
