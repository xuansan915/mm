#include "../reconize/reconize.h"
#include <opencv2/opencv.hpp>

int g_min_face = 200;
int g_pre_min_face = g_min_face;
void onTrackbarslide(int pos,void*)
{
    if(pos < 20)
    {
        g_min_face = 20;
    }
}

int main()
{
    //开启摄像头
    cv::VideoCapture cap(-1);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);//320
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);//240
    int video_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int video_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    std::string  model_path ("../model/facenet_celeb_ncs.graph");
    FaceReconize tFaceReconize(MV_MACHINE);
    tFaceReconize.Init(model_path, 0.7, video_width, video_width,200, 0);

    std::cout << video_width << "   " << video_height << std::endl;

    if (!cap.isOpened())
    {
        std::cout <<"can't open camera "<< std::endl;
        cv::destroyAllWindows();
        cap.release();
        return 0;
    }
    cv::Mat baseFace = cv::imread("../model/face.jpg");
    float baseEigenValue[OUTPUT_LENTH];
    tFaceReconize.GetFaceEigenValue(baseFace, baseEigenValue, sizeof(baseEigenValue));
    float currentEigenValue[OUTPUT_LENTH];

    cv::Mat frame;

    //cap >> frame;
    float fEigenValue[OUTPUT_LENTH];
    cv::Mat detectFace;
    int key = 0;
    int64 starttime = cv::getTickCount();
    int64 endtime = cv::getTickCount();

    while( key != 'q')
    {
        cap >> frame;
        tFaceReconize.CutFaceImage(frame,detectFace);
        if(!detectFace.empty())
        {
                    starttime = cv::getTickCount();

            tFaceReconize.GetFaceEigenValue(detectFace, currentEigenValue, sizeof(baseEigenValue));
            double similarity = tFaceReconize.Match(baseEigenValue, currentEigenValue, OUTPUT_LENTH);
            cv::putText(frame, " similarity1  : "+std::to_string(similarity), cv::Point(50,100), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 250), 1, CV_AVX);
                    endtime = cv::getTickCount();
        	cv::putText(frame, " tFaceReconize1:"+std::to_string((endtime - starttime) / cv::getTickFrequency()), cv::Point(50,150), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 250), 1, CV_AVX);
        }
        endtime = cv::getTickCount();

        cv::putText(frame, " FPS:"+std::to_string(cv::getTickFrequency() / (endtime - starttime)), cv::Point(50,50), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 250), 1, CV_AVX);

        cv::imshow("test", frame);
        starttime = endtime;

        key = cv::waitKey(1);
                cv::createTrackbar("face_size","test",&g_min_face,300,onTrackbarslide);//第五个参数实际是一个数字与进度的对应关系；
        if(g_min_face != g_pre_min_face)
        {
            g_pre_min_face = g_min_face;
            tFaceReconize.ResetDetector(video_width, video_width, g_min_face);
        }
    }
    tFaceReconize.UnInit();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
