#include "mvGraph.h"
#include <iostream>
#define MAX_PATH 256
bool read_graph_from_file(const char *graph_filename, unsigned int *length_read, void **graph_buf)
{
    FILE *graph_file_ptr;

    *graph_buf = nullptr;

    graph_file_ptr = fopen(graph_filename, "rb");
    if (graph_file_ptr == nullptr)
    {
        return false;
    }

    // get number of bytes in file
    *length_read = 0;
    fseek(graph_file_ptr, 0, SEEK_END);
    *length_read = ftell(graph_file_ptr);
    rewind(graph_file_ptr);

    if(!(*graph_buf = malloc(*length_read)))
    {
        // couldn't allocate buffer
        fclose(graph_file_ptr);
        return false;
    }

    size_t to_read = *length_read;
    size_t read_count = fread(*graph_buf, 1, to_read, graph_file_ptr);

    if(read_count != *length_read)
    {
        // didn't read the expected number of bytes
        fclose(graph_file_ptr);
        free(*graph_buf);
        *graph_buf = nullptr;
        return false;
    }
    fclose(graph_file_ptr);

    return true;
}

Graph::Graph(ncGraphHandle_t* api2_graph, ncFifoHandle_t*  fifo_in, ncFifoHandle_t*  fifo_out):
    _api2_graph(api2_graph),_fifo_in(fifo_in),_fifo_out(fifo_out)
{

}

Graph::~Graph()
{

}

void Graph::DealocateGraph()
{
    ncFifoDestroy(&_fifo_in);
    ncFifoDestroy(&_fifo_out);
    ncGraphDestroy(&_api2_graph);
}

void Graph::LoadTensor(void* inputtensor, unsigned int inputTensorLength)
{
    ncStatus_t retCode = ncGraphQueueInferenceWithFifoElem(_api2_graph, _fifo_in, _fifo_out, inputtensor, &inputTensorLength, 0);
    if (retCode != NC_OK)
    {
        // Failed to create the device... maybe it isn't plugged in to the host
        printf("Error - ncGraphQueueInferenceWithFifoElem failed for for device  error %d\n", retCode);
        abort();
    }
}

void Graph::GetResult(void* outputData, unsigned int& outputDataLen)
{
    ncStatus_t retCode = ncFifoReadElem(_fifo_out, outputData, &outputDataLen, 0);
    if (retCode != NC_OK)
    {
        // Failed to create the device... maybe it isn't plugged in to the host
        printf("Error - ncFifoReadElem failed for for device  error %d\n", retCode);
        abort();
    }
}



Device::Device(int deviceIndex):
    _deviceIndex(deviceIndex)
{
    // Initialize the device handle
    ncStatus_t retCode = ncDeviceCreate(deviceIndex, &_api2_device);
    if (retCode != NC_OK)
    {
        // Failed to create the device... maybe it isn't plugged in to the host
        printf("Error - ncDeviceCreate failed for for device at index %d error %d\n", deviceIndex, retCode);
        abort();
    }
}

Device::~Device()
{

}

void Device::OpenDevice()
{
    // Open the device
    ncStatus_t retCode = ncDeviceOpen(_api2_device);
    if (retCode != NC_OK)
    {
        // Failed to open the device
        printf("Error - ncDeviceOpen failed could not open the device at index %d, error: %d.\n", _deviceIndex, retCode);
        ncDeviceDestroy(&_api2_device);
        abort();
    }
}

void Device::CloseDevice()
{
    ncStatus_t retCode = ncDeviceClose(_api2_device);
    if (retCode != NC_OK)
    {
        // Failed to open the device
        printf("Error - ncDeviceClose failed could not close the device at index %d, error: %d.\n", _deviceIndex, retCode);
        ncDeviceDestroy(&_api2_device);
        abort();
    }
    retCode = ncDeviceDestroy(&_api2_device);
    if (retCode != NC_OK)
    {
        // Failed to open the device
        printf("Error - ncDeviceDestroy failed could not distroy the device at index %d, error: %d.\n", _deviceIndex, retCode);
        abort();
    }
}

Graph* Device::AllocateGraph(std::string& graphfile)
{
    ncGraphHandle_t* _api2_graph;
    ncFifoHandle_t*  _fifo_in;
    ncFifoHandle_t*  _fifo_out;
    ncStatus_t status = ncGraphCreate("mvnc_simple_api graph", &_api2_graph);
    unsigned int graph_len = 0;
    void *graph_buf;
    if (!read_graph_from_file(graphfile.c_str(), &graph_len, &graph_buf))
    {
        // error reading graph
        std::cout << "Error - Could not read graph file from disk: " << graphfile << std::endl;
        return nullptr;
    }
    status = ncGraphAllocateWithFifosEx(_api2_device, _api2_graph, graph_buf, graph_len,
                                        &_fifo_in, NC_FIFO_HOST_WO, 2, NC_FIFO_FP32,
                                        &_fifo_out, NC_FIFO_HOST_RO, 2, NC_FIFO_FP32);
    if (status != NC_OK)
    {
        // Failed to open the device
        printf("Error - ncDeviceDestroy failed could not distroy the device at index %d, error: %d.\n", _deviceIndex, status);
        abort();
    }
    return new Graph(_api2_graph, _fifo_in, _fifo_out);
};

void Device::DelocateGraph(Graph* pGraph)
{
    pGraph->DealocateGraph();
    delete pGraph;
}

