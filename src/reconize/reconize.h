#ifndef _FaceReconize_h_
#define _FaceReconize_h_
/*******************************************************************
 *  Copyright(c) 2000-2022 上海无门科技
 *  All rights reserved.
 *
 *  文件名称:
 *  简要描述: 校验输入图片与之前特征值的相似度
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

#include "../eigenvalue/ieigenvalue.h"
enum MachineType
{
    TF_MACHINE,     //tensorflow
    MV_MACHINE      //movidius
};

class FaceReconize
{
public:
    FaceReconize(MachineType eMachineType);
    ~FaceReconize();

public:
    /**
     *  功能描述:   初始化特征提取器
     *  @param strModelPath     模型路径
     *  @param fminSimilarity   特征的最低相似度
     *  @param imageWidth       CompareCurrentFrameWithEigenValue 输入进来的frame 的宽
     *  @param imageHeight      CompareCurrentFrameWithEigenValue 输入进来的frame 的高
     *
     *  @return 是否初始化成功
     */
    bool Init(const std::string &strModelPath, float fminSimilarity,int imageWidth, int imageHeight,int minsize, int index);

    void ResetDetector(int imageWidth, int imageHeight,int minsize);
    /**
     *  功能描述:
     *
     *  @return
     */
    bool UnInit();
    /**
     *  功能描述:   某张图中最大人脸 和 特征值比对   本函数由于涉及到人脸检测，所以避免不必要的调用  以 获得最优的性能
     *  @param frame            某张图
     *  @param fEigenValue      人脸特征点
     *  @param detectFace       图片中检测出来的最大人脸
     *  @return 相似度
     */
    bool CompareCurrentFrameWithEigenValue(cv::Mat &frame, float* fEigenValue,cv::Mat& detectFace);
    /**
    *  功能描述:   裁剪frame，从中取出最大人脸图
    *  @param frame            某张图
    *  @param detectFace       图片中检测出来的最大人脸
    */
    void CutFaceImage(cv::Mat &frame,cv::Mat& detectFace);
    /**
    *  功能描述:   提取人脸图的特征点
    *  @param face             人脸
    *  @param fEigenValue      特征点
    *  @param nOutputLen       特征点长度
    */
    void GetFaceEigenValue(cv::Mat &face, float* fEigenValue, int nOutputLen);
    /**
    *  功能描述:   比对人脸特征点
    *  @param fBaseEigenValue          底图的特征点
    *  @param newBaseEigenValue        当前的特征点
    *  @param nEigenLen                特征点长度
    */
    double Match(float *fBaseEigenValue, float* newBaseEigenValue, int nEigenLen);
    /**
     *  功能描述:   查看这张图所来自的摄像头是否被遮挡
     *  @param frame            某张图
     *  @return 是否遮挡  true 遮挡 false未遮挡
     */
    bool CameraBlocked(cv::Mat &frame);

private:
    //特征提取器
    IEigenValue*       m_pEigenValue;
    //最低的相似度
    float            m_minSimilarity;
};

#endif // _FaceReconize_h_
