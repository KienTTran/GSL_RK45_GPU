#ifndef GPU_FUNCTION_H
#define GPU_FUNCTION_H

#include "gpu_parameters.cuh"
#include "gpu_ode.cuh"
#include "gpu_mcmc.cuh"
#include "gpu_reduce.cuh"

class GPUFlu{
public:
    explicit GPUFlu();
    ~GPUFlu();
    void set_gpu_parameters(GPUParameters* gpu_params);
    void run();
    void init();
private:
    GPUParameters* gpu_params;
    FluParameters* flu_params;
};

#endif
