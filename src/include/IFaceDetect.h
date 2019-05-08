#ifndef IFaceDetect_H
#define IFaceDetect_H
#include <opencv2/opencv.hpp>
#include <vector>
#define mydataFmt float

struct Bbox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool exist;
    mydataFmt ppoint[10];
    mydataFmt regreCoord[4];
};

class IFaceDetect
{
public:
    virtual ~IFaceDetect(){};

    virtual bool Init(int row, int col) = 0;
    virtual void findFace(cv::Mat &image) = 0;

    std::vector<struct Bbox> thirdBbox_;
};

#endif
