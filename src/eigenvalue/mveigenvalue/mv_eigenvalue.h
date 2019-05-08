#ifndef MV_EIGENVALUE_H
#define MV_EIGENVALUE_H
#include "../ieigenvalue.h"

class mv_eigenvalue : public IEigenValue
{
    public:
        mv_eigenvalue();
        virtual ~mv_eigenvalue();

public:
    /**
     *  功能描述:   初始化特征提取器
     *  @param strModelPath     模型路径
     *
     *  @return 是否初始化成功
     */
    virtual bool Init(const std::string &strModelPath,int imageWidth, int imageHeight,int minsize, int index);
    /**
     *  功能描述:
     *
     *  @return
     */
    virtual bool UnInit();
    /**
     *  功能描述:   提取图片中人脸特征值
     *  @param strImagePath     输入图片的路径，只能够包含一个人脸,并且人脸是经过裁剪的
     *
     */
    virtual void GetFaceEigenValue(const std::string &strImagePath, void* outputData, unsigned int outputDataLen);

    /**
     *  功能描述:   提取图片中人脸特征值
     *  @param face     经过裁剪后的人脸图
     *  @param outputData
     *  @param outputDataLen
     *
     *
     */
    virtual void GetFaceEigenValue(cv::Mat &face, void* outputData, unsigned int outputDataLen);

};

#endif // MV_EIGENVALUE_H
