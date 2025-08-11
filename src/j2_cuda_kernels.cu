#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif
#include <cmath>
#include <iostream>
#include "math_constants.h"

#ifndef __CUDACC__
// 当没有CUDA时，提供空的实现
extern "C" {
void cuda_propagate_j2(double* elements, size_t num_satellites, double dt,
                      double mu, double re, double j2) {
    std::cerr << "Warning: CUDA not available. Please use CPU_SCALAR or CPU_SIMD mode." << std::endl;
}

void cuda_compute_positions(double* elements, double* positions, 
                           size_t num_satellites) {
    std::cerr << "Warning: CUDA not available. Please use CPU_SCALAR or CPU_SIMD mode." << std::endl;
}
}
#else

// CUDA设备常数
__constant__ double d_MU = 3.986004418e14;
__constant__ double d_RE = 6378137.0;
__constant__ double d_J2 = 1.08263e-3;

// CUDA设备函数：角度归一化
__device__ double normalize_angle_cuda(double angle) {
    angle = fmod(angle, 2.0 * M_PI);
    if (angle < 0) angle += 2.0 * M_PI;
    return angle;
}

// CUDA内核：J2摄动外推
__global__ void j2_propagate_kernel(double* a, double* e, double* i, 
                                   double* O, double* w, double* M,
                                   int num_satellites, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_satellites) {
        // 加载当前卫星的轨道要素
        double a_val = a[idx];
        double e_val = e[idx];
        double i_val = i[idx];
        double O_val = O[idx];
        double w_val = w[idx];
        double M_val = M[idx];
        
        // 计算平均角速度
        double n = sqrt(d_MU / (a_val * a_val * a_val));
        
        // 计算J2摄动系数
        double one_minus_e2 = 1.0 - e_val * e_val;
        double p = a_val * one_minus_e2;
        double factor = (3.0 * d_J2 * d_MU * d_RE * d_RE) / (2.0 * p * p);
        double common_factor = factor / (n * a_val * a_val);
        
        // 计算三角函数
        double cos_i = cos(i_val);
        double sin_i = sin(i_val);
        double cos2_i = cos_i * cos_i;
        double sin2_i = sin_i * sin_i;
        
        // 计算导数并更新轨道要素
        double dO = -common_factor * cos_i * dt;
        double dw = common_factor * (2.0 - 2.5 * sin2_i) * dt;
        double dM_term = factor * ((3.0 * cos2_i - 1.0) / 2.0) * one_minus_e2 / (n * a_val * a_val);
        double dM = (n + dM_term) * dt;
        
        // 更新并归一化角度
        O[idx] = normalize_angle_cuda(O_val + dO);
        w[idx] = normalize_angle_cuda(w_val + dw);
        M[idx] = normalize_angle_cuda(M_val + dM);
    }
}

// CUDA内核：批量位置计算
__global__ void compute_positions_kernel(double* a, double* e, double* i,
                                        double* O, double* w, double* M,
                                        double* pos_x, double* pos_y, double* pos_z,
                                        int num_satellites) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_satellites) {
        double a_val = a[idx];
        double e_val = e[idx];
        double i_val = i[idx];
        double O_val = O[idx];
        double w_val = w[idx];
        double M_val = M[idx];
        
        // 简化的偏近点角计算 (牛顿法1次迭代)
        double E = M_val;  // 初始猜测
        E = E - (E - e_val * sin(E) - M_val) / (1.0 - e_val * cos(E));
        
        // 真近点角
        double tan_nu_2 = sqrt((1.0 + e_val) / (1.0 - e_val)) * tan(E / 2.0);
        double nu = 2.0 * atan(tan_nu_2);
        
        // 地心距
        double r = a_val * (1.0 - e_val * cos(E));
        
        // 轨道平面内位置
        double x_perifocal = r * cos(nu);
        double y_perifocal = r * sin(nu);
        
        // 转换矩阵元素
        double cosO = cos(O_val);
        double sinO = sin(O_val);
        double cosi = cos(i_val);
        double sini = sin(i_val);
        double cosw = cos(w_val);
        double sinw = sin(w_val);
        
        // 转换到惯性系
        pos_x[idx] = (cosO*cosw - sinO*sinw*cosi) * x_perifocal + 
                     (-cosO*sinw - sinO*cosw*cosi) * y_perifocal;
        pos_y[idx] = (sinO*cosw + cosO*sinw*cosi) * x_perifocal + 
                     (-sinO*sinw + cosO*cosw*cosi) * y_perifocal;
        pos_z[idx] = sinw*sini * x_perifocal + cosw*sini * y_perifocal;
    }
}

// C接口函数
extern "C" {
    void cuda_propagate_j2(double* elements, size_t num_satellites, double dt, 
                          double mu, double re, double j2) {
        
        // 设置设备常数
        cudaMemcpyToSymbol(d_MU, &mu, sizeof(double));
        cudaMemcpyToSymbol(d_RE, &re, sizeof(double));
        cudaMemcpyToSymbol(d_J2, &j2, sizeof(double));
        
        // 分配设备内存
        double *d_a, *d_e, *d_i, *d_O, *d_w, *d_M;
        size_t size = num_satellites * sizeof(double);
        
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_e, size);
        cudaMalloc(&d_i, size);
        cudaMalloc(&d_O, size);
        cudaMalloc(&d_w, size);
        cudaMalloc(&d_M, size);
        
        // 复制数据到设备 (假设elements按SoA格式存储)
        cudaMemcpy(d_a, elements, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_e, elements + num_satellites, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_i, elements + 2*num_satellites, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_O, elements + 3*num_satellites, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, elements + 4*num_satellites, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_M, elements + 5*num_satellites, size, cudaMemcpyHostToDevice);
        
        // 计算网格和块尺寸
        int block_size = 256;
        int grid_size = (num_satellites + block_size - 1) / block_size;
        
        // 启动内核
        j2_propagate_kernel<<<grid_size, block_size>>>(
            d_a, d_e, d_i, d_O, d_w, d_M, num_satellites, dt);
        
        // 检查错误
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA kernel error: %s\n", cudaGetErrorString(error));
        }
        
        // 复制结果回主机
        cudaMemcpy(elements + 3*num_satellites, d_O, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(elements + 4*num_satellites, d_w, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(elements + 5*num_satellites, d_M, size, cudaMemcpyDeviceToHost);
        
        // 清理设备内存
        cudaFree(d_a);
        cudaFree(d_e);
        cudaFree(d_i);
        cudaFree(d_O);
        cudaFree(d_w);
        cudaFree(d_M);
    }
    
    void cuda_compute_positions(double* elements, double* positions, 
                               size_t num_satellites) {
        // 分配设备内存
        double *d_a, *d_e, *d_i, *d_O, *d_w, *d_M;
        double *d_pos_x, *d_pos_y, *d_pos_z;
        size_t size = num_satellites * sizeof(double);
        
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_e, size);
        cudaMalloc(&d_i, size);
        cudaMalloc(&d_O, size);
        cudaMalloc(&d_w, size);
        cudaMalloc(&d_M, size);
        cudaMalloc(&d_pos_x, size);
        cudaMalloc(&d_pos_y, size);
        cudaMalloc(&d_pos_z, size);
        
        // 复制轨道要素到设备
        cudaMemcpy(d_a, elements, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_e, elements + num_satellites, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_i, elements + 2*num_satellites, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_O, elements + 3*num_satellites, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, elements + 4*num_satellites, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_M, elements + 5*num_satellites, size, cudaMemcpyHostToDevice);
        
        // 计算网格和块尺寸
        int block_size = 256;
        int grid_size = (num_satellites + block_size - 1) / block_size;
        
        // 启动内核
        compute_positions_kernel<<<grid_size, block_size>>>(
            d_a, d_e, d_i, d_O, d_w, d_M, d_pos_x, d_pos_y, d_pos_z, num_satellites);
        
        // 复制位置结果回主机
        cudaMemcpy(positions, d_pos_x, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(positions + num_satellites, d_pos_y, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(positions + 2*num_satellites, d_pos_z, size, cudaMemcpyDeviceToHost);
        
        // 清理设备内存
        cudaFree(d_a); cudaFree(d_e); cudaFree(d_i);
        cudaFree(d_O); cudaFree(d_w); cudaFree(d_M);
        cudaFree(d_pos_x); cudaFree(d_pos_y); cudaFree(d_pos_z);
    }
}

#endif // __CUDACC__