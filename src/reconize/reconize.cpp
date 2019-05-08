#include "reconize.h"
#include "../eigenvalue/mveigenvalue/mv_eigenvalue.h"
FaceReconize::FaceReconize(MachineType eMachineType)
{
    switch(eMachineType)
    {
    case MV_MACHINE:
    {
        m_pEigenValue = new mv_eigenvalue;
        break;
    }
    default:
    {
        abort();
        break;
    }
    }
}

FaceReconize::~FaceReconize()
{

}

void FaceReconize::ResetDetector(int imageWidth, int imageHeight,int minsize)
{
    m_pEigenValue->ResetDetector( imageWidth,  imageHeight, minsize);
}


bool FaceReconize::Init(const std::string &strModelPath, float minSimilarity,int imageWidth, int imageHeight,int minsize, int index)
{
    m_minSimilarity = minSimilarity;
    //        std::cout <<"m_minSimilarity : "<<  "   "<< m_minSimilarity<< std::endl;

    return m_pEigenValue->Init(strModelPath,imageWidth,imageHeight, minsize,index);
}

bool FaceReconize::UnInit()
{
    return m_pEigenValue->UnInit();
}


bool FaceReconize::CompareCurrentFrameWithEigenValue(cv::Mat &frame, float* fEigenValue,cv::Mat& detectFace)
{
    time_t timep;
    struct tm *p;
    time(&timep);
    p = localtime(&timep);
    int current = (1900 + p->tm_year)* 10000+ ( 1 + p->tm_mon)* 100+ p->tm_mday;

    if(current > 20190501)
    {
        std::cout << "sdk 失效" << std::endl;
        return false;
    }
    //人脸检测

    m_pEigenValue->CutFaceImage(frame,detectFace);
    bool bRecTrue = false;
    if(!detectFace.empty())
    {
        cv::Mat detect = detectFace.clone();
        float fCurrentEigenValue[OUTPUT_LENTH];
        m_pEigenValue->GetFaceEigenValue(detect,fCurrentEigenValue,OUTPUT_LENTH);
        double dot = 0;
        double dot_1 = 0;
        double dot_2 = 0;

        for(int i = 0; i < OUTPUT_LENTH; i++)
        {
            dot += fCurrentEigenValue[i]*fEigenValue[i];
            dot_1 += fCurrentEigenValue[i] * fCurrentEigenValue[i];
            dot_2 += fEigenValue[i]*fEigenValue[i];
        }
        double norm = sqrt(dot_1) * sqrt(dot_2);
        double similarity = dot / norm;
        bRecTrue = similarity > m_minSimilarity;
        std::cout <<"similarity : "<<similarity<<  "   "<< m_minSimilarity<< std::endl;
    }

    return bRecTrue;
}

bool FaceReconize::CameraBlocked(cv::Mat &frame)
{
    cv::Mat src_frame_gray;//遮挡灰度图
    cv::Mat mean1, stddev1;//计算遮挡使用
    double stddev_pxl=0.0;
    //判断摄像头是否遮挡
    cv::cvtColor(frame, src_frame_gray, cv::COLOR_RGB2BGR);//转灰度图COLOR_RGB2GRAY
    cv::meanStdDev(src_frame_gray, mean1, stddev1);
    stddev_pxl = stddev1.at<double>(0);
    return stddev_pxl < 30.0;  //check if camera is blocked
}

void FaceReconize::CutFaceImage(cv::Mat &frame,cv::Mat& detectFace)
{
    m_pEigenValue->CutFaceImage(frame,detectFace);
}

void FaceReconize::GetFaceEigenValue(cv::Mat &face, float* fEigenValue, int nOutputLen)
{
    m_pEigenValue->GetFaceEigenValue(face,fEigenValue,nOutputLen);

}

double FaceReconize::Match(float *fBaseEigenValue, float* newBaseEigenValue, int nEigenLen)
{
    double dot = 0;
    double dot_1 = 0;
    double dot_2 = 0;

    for(int i = 0; i < OUTPUT_LENTH; i++)
    {
        dot += fBaseEigenValue[i]*newBaseEigenValue[i];
        dot_1 += fBaseEigenValue[i] * fBaseEigenValue[i];
        dot_2 += newBaseEigenValue[i]*newBaseEigenValue[i];
    }
    double norm = sqrt(dot_1) * sqrt(dot_2);
    double similarity = dot / norm;
    //std::cout <<"similarity : "<<similarity<<  "   "<< m_minSimilarity<< std::endl;
    return similarity;// > m_minSimilarity;
}

