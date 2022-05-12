#ifndef GPU_FLU_STREAM_H
#define GPU_FLU_STREAM_H

#include "../gpu_parameters.cuh"
#include "gpu_ode_stream.cuh"
#include "../gpu_mcmc.cuh"

class GPUStreamFlu{
public:
    explicit GPUStreamFlu();
    ~GPUStreamFlu();
    void set_gpu_parameters(GPUParameters* gpu_params);
    void run();
    void init();
private:
    GPUParameters* gpu_params;
    FluParameters* flu_params;
};

#endif
