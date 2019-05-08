#include "ieigenvalue.h"
#include <math.h>

IEigenValue::IEigenValue():
    m_pFaceDetect(nullptr),
    m_pIReasonMachine(nullptr)
{

}

IEigenValue::~IEigenValue()
{
}

void IEigenValue::CutFaceImage(cv::Mat &source,cv::Mat &maxface, bool bJustDetect)
{
    m_pFaceDetect->thirdBbox_.clear();
    //人脸检测
    m_pFaceDetect->findFace(source);

    int old_width = 0;
    cv::Rect faceRect = cv::Rect(0,0,0,0);
    bool bFind = false;

    for(std::vector<struct Bbox>::iterator it = m_pFaceDetect->thirdBbox_.begin(); it != m_pFaceDetect->thirdBbox_.end(); it++)
    {
        if((*it).exist)
        {
            if(bJustDetect)
            {
                cv::rectangle(source, cv::Point((*it).y1, (*it).x1), cv::Point((*it).y2, (*it).x2), cv::Scalar(0,0,255), 2,8,0);

            }
            else
            {
                int new_width = (*it).x2-(*it).x1;
                if (new_width > old_width)
                {
                    bFind = true;
                    old_width = new_width;
                    faceRect = cv::Rect((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
                    faceRect.x = faceRect.x + faceRect.width * 0.1;
                    faceRect.width = faceRect.width - faceRect.width * 2 * 0.1;
                }
            }
        }
    }

    if(bFind)
    {
        cv::rectangle(source, cv::Point(faceRect.x, faceRect.y), cv::Point(faceRect.x + faceRect.width, faceRect.y + faceRect.height ), cv::Scalar(0,0,255), 2,8,0);

        maxface = source(faceRect).clone();
    }
    else if(!maxface.empty())
    {
        //检测不到就释放掉
        maxface.release();
    }
}
