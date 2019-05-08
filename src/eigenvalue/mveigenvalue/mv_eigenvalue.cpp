#include "mv_eigenvalue.h"
#include "../../reasoningmachine/mvreasonmachine/mv_reasonmachine.h"
#include "../../mtcnn/mtcnn.h"
mv_eigenvalue::mv_eigenvalue():
    IEigenValue()

{
    //ctor
}

mv_eigenvalue::~mv_eigenvalue()
{
    //dtor
}

bool mv_eigenvalue::Init(const std::string &strModelPath,int imageWidth, int imageHeight, int index)
{
    m_pFaceDetect = new mtcnn;
    m_pIReasonMachine = new mv_reasonmachine(strModelPath, index, NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT);
    return m_pIReasonMachine->Init() && m_pFaceDetect->Init(imageWidth,imageHeight);
}

bool mv_eigenvalue::UnInit()
{
    m_pIReasonMachine->UnInit();
    delete m_pIReasonMachine;
}

void mv_eigenvalue::GetFaceEigenValue(const std::string &strImagePath, void* outputData, unsigned int outputDataLen)
{
    cv::Mat face = cv::imread(strImagePath);
    if(face.empty())
    {
        std::cout << "获取图片失败:  " << strImagePath << std::endl;
        return ;
    }
    cv::Mat maxface;
    CutFaceImage(face, maxface);
    m_pIReasonMachine->ReasonImage(maxface);
    m_pIReasonMachine->GetReasonResult(outputData, outputDataLen);
}

void mv_eigenvalue::GetFaceEigenValue(cv::Mat &face, void* outputData, unsigned int outputDataLen)
{
    m_pIReasonMachine->ReasonImage(face);
    m_pIReasonMachine->GetReasonResult(outputData, outputDataLen);
}
