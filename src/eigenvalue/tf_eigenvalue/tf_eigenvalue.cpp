#include "tf_eigenvalue.h"

#include "../TFRunner/TFRunner.h"
#include "../mtcnn/mtcnn.h"
std::string inputname("input:0");
std::string outputName("embeddings:0");
int inputheigt = 160;
int inputwidth = 160;
int inputchannel = 3;
bool isRGB = true;

tf_eigenvalue::tf_eigenvalue():
    m_pITFRunner(nullptr),
    m_pFaceDetect(new mtcnn)

{

}

tf_eigenvalue::~tf_eigenvalue()
{
    if(m_pFaceDetect != nullptr)
        delete m_pFaceDetect;
}

bool tf_eigenvalue::Init(const std::string &strModelPath,int imageWidth, int imageHeight)
{

    m_pITFRunner = new TFRunner(const_cast<std::string &>(strModelPath),inputname,outputName,inputheigt,inputwidth,inputchannel,isRGB);
    return m_pITFRunner->Init() && m_pFaceDetect->Init(imageWidth,imageHeight);
}

bool tf_eigenvalue::UnInit()
{
    if(m_pITFRunner != nullptr)
        delete m_pITFRunner;
    return true;

}

float* tf_eigenvalue::GetFaceEigenValue(const std::string &strImagePath)
{
    cv::Mat source = cv::imread(strImagePath);
    if(source.empty())
    {
        std::cout << "获取图片失败:  " << strImagePath << std::endl;
        return nullptr;
    }
    cv::Mat maxface;
    CutFaceImage(source, maxface);
    return GetFaceEigenValue(maxface);
}

float* tf_eigenvalue::GetFaceEigenValue(cv::Mat &face)
{
    std::vector<tensorflow::Tensor> outputs;

    if(!m_pITFRunner->inference(face, outputs))
    {
        std::cout << "推理失败:  " << std::endl;
        return nullptr;
    }

    float *res = new float[512];
    auto outMap = outputs[0].tensor<float, 2>();
    if(res != nullptr)
    {
        for(int i = 0; i < 512; i++)
            res[i] = outMap(i);
    }

    //face.release();

    return res;
}

void tf_eigenvalue::CutFaceImage(cv::Mat &source,cv::Mat &maxface)
{
    //人脸检测
    m_pFaceDetect->findFace(source);

    int old_width = 0;
    cv::Rect faceRect = cv::Rect(0,0,0,0);
    bool bFind = false;
    for(std::vector<struct Bbox>::iterator it=m_pFaceDetect->thirdBbox_.begin(); it!=m_pFaceDetect->thirdBbox_.end(); it++)
    {
        if((*it).exist)
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

    m_pFaceDetect->thirdBbox_.clear();
    if(bFind)
    {
        maxface = source(faceRect).clone();
    }
}

