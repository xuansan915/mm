#ifndef MV_REASONMACHINE_H
#define MV_REASONMACHINE_H
#include "../IReasoningMachine.h"
#include "mvGraph.h"

class mv_reasonmachine : public IReasoningMachine
{
public:
    mv_reasonmachine(std::string  model_path, int index, int nNetWorkWidth, int nNetWorkHeight);
    virtual ~mv_reasonmachine();

public:
    virtual bool Init() ;
    virtual bool UnInit();
    virtual bool ReasonImage(cv::Mat &images);
    virtual void GetReasonResult(void* outputData, unsigned int outputDataLen);
//private:
//    void preprocess_image(const cv::Mat& src_image_mat, cv::Mat& preprocessed_image_mat);

private:
    Device m_tDevice;
    Graph * m_pGraph;
};

#endif // MV_REASONMACHINE_H
