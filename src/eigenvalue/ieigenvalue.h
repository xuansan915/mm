#ifndef _eigenvalue_h_
#define _eigenvalue_h_
/*******************************************************************
 *  Copyright(c) 2000-2022 上海无门科技
 *  All rights reserved.
 *
 *  文件名称:
 *  简要描述: 提取输入底图的特征值，用于后续抓取到的人脸进行特征值比对
 *
 *  创建日期:
 *  作者:
 *  说明:
 *
 *  修改日期:
 *  作者:
 *  说明:
 ******************************************************************/

#include <string>

#include "../reasoningmachine/IReasoningMachine.h"
#include "../include/IFaceDetect.h"
/*
#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
*/
#define OUTPUT_LENTH 512

#define NETWORK_IMAGE_WIDTH  160
#define NETWORK_IMAGE_HEIGHT 160

class IEigenValue
{
public:
    IEigenValue();
    virtual ~IEigenValue();

public:
    /**
     *  功能描述:   初始化特征提取器
     *  @param strModelPath     模型路径
     *
     *  @return 是否初始化成功
     */
    virtual bool Init(const std::string &strModelPath,int imageWidth, int imageHeight, int index) = 0;
    /**
     *  功能描述:
     *
     *  @return
     */
    virtual bool UnInit() = 0;
    /**
     *  功能描述:   提取图片中人脸特征值
     *  @param strImagePath     输入图片的路径，只能够包含一个人脸,并且人脸是经过裁剪的
     *
     */
    virtual void GetFaceEigenValue(const std::string &strImagePath, void* outputData, unsigned int outputDataLen) = 0;

    /**
     *  功能描述:   提取图片中人脸特征值
     *  @param face     经过裁剪后的人脸图
     *  @param outputData
     *  @param outputDataLen
     *
     *
     */
    virtual void GetFaceEigenValue(cv::Mat &face, void* outputData, unsigned int outputDataLen) = 0;

    /**
     *  功能描述:     裁剪后的人脸图，获得其中最大的一个人脸图
     *  @param source     原始人脸图
     *  @param maxface     原始人脸图中最大的人脸图
     *
     *  @return
     */
    void CutFaceImage(cv::Mat &source,cv::Mat &maxface, bool bJustDetect = false);
protected:
    IReasoningMachine       *m_pIReasonMachine;
    IFaceDetect             *m_pFaceDetect;
    std::vector<float>      m_closed_eye_coef_left;
    std::vector<float>      m_closed_eye_coef_right;
              //  dlib::shape_predictor pose_model_eyelid;

};

#endif // _eigenvalue_h_
