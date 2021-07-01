#include "common.hpp"

int CUDASelector::operator()(const device &Device) {
    const std::string DriverVersion = Device.get_info<info::device::driver_version>();

    if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos))
        return 1;

    return 0;
}


int IntelGPUSelector::operator()(const device &Device) {
    const std::string vendor = Device.get_info<info::device::vendor>();

    if (Device.is_gpu() && (vendor.find("Intel(R) Corporation") != std::string::npos)) {
        gpu_counter++;

        if(gpu_counter > gpus_taken) {
            gpu_counter = 0;
            gpus_taken++;
            return 1;
        }
    }

    return 0;
}