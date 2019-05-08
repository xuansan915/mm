#ifndef MVGRAPH_H
#define MVGRAPH_H
#include <mvnc.h>
#include <string>
class Graph
{
public:
    Graph(ncGraphHandle_t* _api2_graph, ncFifoHandle_t*  _fifo_in, ncFifoHandle_t*  _fifo_out);
    ~Graph();
public:
    void DealocateGraph();
    void LoadTensor(void* inputtensor, unsigned int inputTensorLength);
    void GetResult(void* outputData, unsigned int& outputDataLen);

private:
    ncGraphHandle_t* _api2_graph;
    ncFifoHandle_t*  _fifo_in;
    ncFifoHandle_t*  _fifo_out;
};


class Device
{
public:
    Device(int deviceIndex);
    ~Device();
public:
    void OpenDevice();
    void CloseDevice();

    //void SetDeviceOption();
    Graph* AllocateGraph(std::string& graphfile);
    void DelocateGraph(Graph* pGraph);
private:
    int                 _deviceIndex;
    ncDeviceHandle_t*   _api2_device;
};
#endif // MVGRAPH_H
