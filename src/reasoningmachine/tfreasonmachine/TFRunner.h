#ifndef TFRUNNER_H
#define TFRUNNER_H

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/lib/io/path.h"

#include "../include/ITFRunner.h"


class TFRunner : public ITFRunner
{
public:
    TFRunner(std::string &modelFile,std::string& inputname,std::string& outputName,int inputheigt,int inputwidth,int inputchannel,bool isRGB);
     ~TFRunner();

public:
     virtual bool Init();

     virtual bool inference( cv::Mat &img,std::vector<tensorflow::Tensor>& outputs);

     virtual bool UnInit();
protected:
    std::string& m_modelFile;
    std::string& m_inputname;
    std::vector<std::string> m_outputLayers;
    std::unique_ptr<tensorflow::Session> session;

    std::vector<std::pair<std::string,tensorflow::Tensor>> input_layers;
    bool        m_isRGB;
    cv::Mat     m_Image;
};

#endif // TFRUNNER_H
