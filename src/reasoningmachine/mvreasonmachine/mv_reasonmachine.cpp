#include "mv_reasonmachine.h"
#include <vector>
mv_reasonmachine::mv_reasonmachine(std::string  model_path, int index, int nNetWorkWidth, int nNetWorkHeight):
    m_tDevice(index),IReasoningMachine(model_path, nNetWorkWidth, nNetWorkHeight)
{

}

mv_reasonmachine::~mv_reasonmachine()
{

}

bool mv_reasonmachine::Init()
{
    m_tDevice.OpenDevice();
    m_pGraph = m_tDevice.AllocateGraph(m_model_path);
    return true;
}

bool mv_reasonmachine::UnInit()
{
    m_tDevice.DelocateGraph(m_pGraph);
    m_tDevice.CloseDevice();
    return true;
}

bool  mv_reasonmachine::ReasonImage(cv::Mat &images)
{
    cv::Mat preprocessed_image;

    preprocess_image(images, preprocessed_image);
    int height = preprocessed_image.rows;
    int width = preprocessed_image.cols;
    int depth = preprocessed_image.channels();

    int tensor_index = 0;
    std::vector<float> tensor16(m_nNetWorkWidth * m_nNetWorkHeight * 3);
    tensor16.clear();
    //float tensor16[m_nNetWorkWidth * m_nNetWorkHeight * 3];
    //std::cout << "height " << height << " width "<< width << "  depth " << depth<< std::endl;
    // copying the data into the corresponding tensor
    for (int y = 0; y < height; ++y)
    {
        const double* source_row = preprocessed_image.ptr<double>(y);
        for (int x = 0; x < width; ++x)
        {
            const double* source_pixel = source_row + (x * depth);
            for (int c = 0; c < depth; ++c)
            {
                const double* source_value = source_pixel + c;//RGB->BGR
                tensor16[tensor_index++] =  (*source_value);
            }
        }
    }
    unsigned int inputTensorLength = m_nNetWorkWidth * m_nNetWorkHeight * 3 * sizeof(float);

    m_pGraph->LoadTensor(tensor16.data(), inputTensorLength);
}

void mv_reasonmachine::GetReasonResult(void* outputData, unsigned int outputDataLen)
{
    m_pGraph->GetResult(outputData, outputDataLen);
}
