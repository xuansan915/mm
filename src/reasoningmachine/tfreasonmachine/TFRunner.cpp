#include "TFRunner.h"
#include <math.h>
#include "./Logger.h"



void splitStringByIcon( std::string& strSource, std::string& strSplitor, std::vector<std::string>& spliteList)
{
    std::string::size_type pos = strSource.find(strSplitor);
    std::string::size_type lastpos = 0;
    while(pos != -1)
    {
        std::string subStr = strSource.substr(lastpos,pos);
        strSource = strSource.substr(pos + 1);
        spliteList.emplace_back(subStr);
        pos = strSource.find(strSplitor);
    }

    spliteList.emplace_back(strSource);
}

TFRunner::TFRunner(std::string &modelFile,std::string& inputname,std::string& outputName,int inputheigt,int inputwidth,int inputchannel,bool isRGB):
    m_modelFile(modelFile),m_inputname(inputname),input_layers(1),m_Image(cv::Size(inputheigt, inputwidth), CV_8UC(inputchannel)),m_isRGB(isRGB)

{
//    initLogger("/home/jerry/a.txt");
    std::string splitor(",");
    splitStringByIcon(outputName, splitor, m_outputLayers);
}

TFRunner::~TFRunner()
{
    //dtor
}

bool TFRunner::Init()
{
    //tensorflow::Tensor tensor_a_1(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, network_height, network_width, network_depth}));
    std::cout << m_modelFile <<std::endl;

    tensorflow::GraphDef graph_def;
    if (!ReadBinaryProto(tensorflow::Env::Default(), m_modelFile, &graph_def).ok())
    {
        std::cout << "Read proto  error" <<std::endl;
        return false;
    }
    tensorflow::SessionOptions sess_opt;
    //sess_opt.config.mutable_gpu_options()->set_allow_growth(true);
    tensorflow::ConfigProto* config = &sess_opt.config;
    //config->set_inter_op_parallelism_threads(8);
    //config->set_intra_op_parallelism_threads(8);
    //config->mutable_gpu_options()->set_allow_growth(true);
    //config->set_allow_soft_placement(true);
    //config->mutable_gpu_options()->set_visible_device_list("0");


    (&session)->reset(tensorflow::NewSession(sess_opt));

    if (!session->Create(graph_def).ok())
    {
        std::cout << "Create graph" <<std::endl;
        return false;
    }
    std::cout << "Init success" <<std::endl;
    tensorflow::Tensor input_tensor = tensorflow::Tensor(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({1, m_Image.rows, m_Image.cols,m_Image.channels()}));

    // 5. 创建输入输出tensor
    input_layers.clear();
    input_layers.push_back({m_inputname, input_tensor});
    tensorflow::Tensor phase_train(tensorflow::DT_BOOL, tensorflow::TensorShape());
    phase_train.scalar<bool>()() = false;

    std::string input_layer_2 = "phase_train:0";
    input_layers.push_back({input_layer_2, phase_train});

    std::vector<tensorflow::Tensor> outputs(1);


    // 7. 运行模型，需要传递输入tensor，输出tensor，输出tensor的name ---softmax
    tensorflow::Status status = session->Run(input_layers, m_outputLayers, {}, &outputs);
    if(!status.ok())
    {
        std::cout << "推理出错!!!!!!" <<std::endl;
        return false;
    }
    return true;
}

//#define Time


void getImageTensor(tensorflow::Tensor &input_tensor, cv::Mat& Image, int height, int width, int depth)
{
    if(Image.cols != width || Image.rows != height)
    {
        resize(Image, Image, cv::Size(width, height));
    }
    //int64_t start = cv::getTickCount();
    // cv::Mat Image = cv::imread(path);
    auto input_tensor_mapped = input_tensor.tensor<float, 4>();

    //mean and std
    //c * r * 3 => c * 3r * 1
    cv::Mat temp = Image.reshape(1, Image.rows * 3);

    cv::Mat mean3;
    cv::Mat stddev3;
    cv::meanStdDev(temp, mean3, stddev3);

    double mean_pxl = mean3.at<double>(0);
    double stddev_pxl = stddev3.at<double>(0);

    //prewhiten
    Image.convertTo(Image, CV_64FC3);
    Image = Image - cv::Vec3d(mean_pxl, mean_pxl, mean_pxl);
    Image = Image / stddev_pxl;

    // copying the data into the corresponding tensor
    for (int y = 0; y < height; ++y)
    {
        const double* source_row = Image.ptr<double>(y);
        for (int x = 0; x < width; ++x)
        {
            const double* source_pixel = source_row + (x * depth);
            for (int c = 0; c < depth; ++c)
            {
                const double* source_value = source_pixel + (2-c);//RGB->BGR
                input_tensor_mapped(0, y, x, c) = *source_value;
            }
        }
    }
    mean3.release();
    stddev3.release();
}


bool TFRunner::inference( cv::Mat &image,std::vector<tensorflow::Tensor>& outputs)
{
    tensorflow::Tensor input_tensor = tensorflow::Tensor(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({1, m_Image.rows, m_Image.cols,m_Image.channels()}));
    //float *pTensor = input_tensor.flat<float>().data();

    getImageTensor(input_tensor,image,m_Image.rows, m_Image.cols,m_Image.channels());

    // 5. 创建输入输出tensor
    input_layers.clear();
    input_layers.push_back({m_inputname, input_tensor});
    tensorflow::Tensor phase_train(tensorflow::DT_BOOL, tensorflow::TensorShape());
    phase_train.scalar<bool>()() = false;

    std::string input_layer_2 = "phase_train:0";
    input_layers.push_back({input_layer_2, phase_train});

    // 7. 运行模型，需要传递输入tensor，输出tensor，输出tensor的name ---softmax
    tensorflow::Status status = session->Run(input_layers, m_outputLayers, {}, &outputs);
    if(!status.ok())
    {
        std::cout << "推理出错!!!!!!" <<std::endl;
        return false;
    }
    return true;
}

bool TFRunner::UnInit()
{
    return true;
}

