// Copyright 2024 The Trae Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file j2_cuda_kernels.cu
 * @brief 包含用于J2摄动轨道外推和位置计算的CUDA内核和接口函数。
 *
 * 该文件为支持CUDA的GPU提供了J2摄动模型的核心计算逻辑。
 * 它定义了两个主要的CUDA内核：
 * 1. `j2_propagate_kernel`: 并行更新大量卫星的长期轨道要素（升交点赤经、近地点幅角、平近点角）。
 * 2. `compute_positions_kernel`: 将轨道要素转换为惯性系中的笛卡尔坐标。
 *
 * 文件还提供了C风格的外部接口函数，以便从C++代码中调用这些CUDA内核。
 * 如果编译时未启用CUDA (`__CUDACC__` 未定义)，则会提供这些接口的空实现，并打印警告信息。
 *
 * 数据布局：
 * 为了在GPU上实现高效的内存访问（合并访问），轨道要素和位置数据采用“结构数组”(SoA)的布局方式。
 * 例如，一个包含N个卫星的数组`elements`在内存中布局如下：
 * [a_1, a_2, ..., a_N, e_1, e_2, ..., e_N, i_1, ..., M_N]
 * 同样，`positions`数组布局为：
 * [x_1, x_2, ..., x_N, y_1, y_2, ..., y_N, z_1, ..., z_N]
 */

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif
#include <cmath>
#include <iostream>
#include "math_constants.h"

// 当没有CUDA时，提供空的实现，以确保代码可以链接和编译。
#ifndef __CUDACC__
extern "C" {
/**
 * @brief J2轨道外推的CUDA接口（空实现）。
 * @param elements 指向轨道要素数据数组的指针（SoA布局）。
 * @param num_satellites 卫星数量。
 * @param dt 时间步长（秒）。
 * @param mu 地球引力常数。
 * @param re 地球赤道半径。
 * @param j2 地球J2摄动系数。
 */
void cuda_propagate_j2(double* elements, size_t num_satellites, double dt,
                      double mu, double re, double j2) {
    // 如果在没有CUDA支持的情况下调用此函数，则向标准错误流打印警告。
    std::cerr << "Warning: CUDA not available. Please use CPU_SCALAR or CPU_SIMD mode." << std::endl;
}

/**
 * @brief 从轨道要素计算位置的CUDA接口（空实现）。
 * @param elements 指向轨道要素数据数组的指针（SoA布局）。
 * @param positions 指向存储计算出的位置坐标的数组指针（SoA布局）。
 * @param num_satellites 卫星数量。
 */
void cuda_compute_positions(double* elements, double* positions, 
                           size_t num_satellites) {
    // 如果在没有CUDA支持的情况下调用此函数，则向标准错误流打印警告。
    std::cerr << "Warning: CUDA not available. Please use CPU_SCALAR or CPU_SIMD mode." << std::endl;
}
}
#else

// 使用__constant__内存来存储全局物理参数。
// 这允许GPU上的所有线程高效地访问这些只读值。
__constant__ double d_MU = 3.986004418e14; ///< 地球引力常数 (m^3/s^2)
__constant__ double d_RE = 6378137.0;     ///< 地球赤道半径 (m)
__constant__ double d_J2 = 1.08263e-3;     ///< 地球J2摄动系数

/**
 * @brief 将角度归一化到 [0, 2*PI) 范围内。
 * @param angle 要归一化的角度（弧度）。
 * @return 归一化后的角度（弧度）。
 */
__device__ double normalize_angle_cuda(double angle) {
    angle = fmod(angle, 2.0 * CUDART_PI );
    if (angle < 0) {
        angle += 2.0 *CUDART_PI;
    }
    return angle;
}

/**
 * @brief J2摄动外推的CUDA内核。
 *
 * 该内核为每个卫星启动一个线程，并行计算由于J2摄动引起的轨道要素的长期变化。
 * 它只更新升交点赤经(O)、近地点幅角(w)和平近点角(M)，因为半长轴、偏心率和倾角在J2模型下是长期不变的。
 *
 * @param a 指向半长轴数组的设备指针。
 * @param e 指向偏心率数组的设备指针。
 * @param i 指向倾角数组的设备指针。
 * @param O 指向升交点赤经数组的设备指针（输入/输出）。
 * @param w 指向近地点幅角数组的设备指针（输入/输出）。
 * @param M 指向平近点角数组的设备指针（输入/输出）。
 * @param num_satellites 卫星总数。
 * @param dt 时间步长（秒）。
 */
__global__ void j2_propagate_kernel(double* a, double* e, double* i, 
                                   double* O, double* w, double* M,
                                   int num_satellites, double dt) {
    // 计算当前线程处理的卫星索引。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_satellites) {
        // 从全局内存加载当前卫星的轨道要素到寄存器，以加快访问速度。
        double a_val = a[idx];
        double e_val = e[idx];
        double i_val = i[idx];
        double O_val = O[idx];
        double w_val = w[idx];
        double M_val = M[idx];
        
        // 计算平均角速度 n = sqrt(mu / a^3)
        double n = sqrt(d_MU / (a_val * a_val * a_val));
        
        // 计算J2摄动引起的长期变化率的公共因子。
        // p = a * (1 - e^2)
        double one_minus_e2 = 1.0 - e_val * e_val;
        double p = a_val * one_minus_e2;
        // factor = (3/2) * J2 * mu * Re^2 / p^2
        double factor = (3.0 * d_J2 * d_MU * d_RE * d_RE) / (2.0 * p * p);
        double common_factor = factor / (n * a_val * a_val);
        
        // 预计算三角函数值。
        double cos_i = cos(i_val);
        double sin_i = sin(i_val);
        double cos2_i = cos_i * cos_i;
        double sin2_i = sin_i * sin_i;
        
        // 计算升交点赤经、近地点幅角和平近点角的导数，并乘以时间步长得到变化量。
        // dO/dt = - (3/2) * n * J2 * (Re/p)^2 * cos(i)
        double dO = -common_factor * cos_i * dt;
        // dw/dt = (3/2) * n * J2 * (Re/p)^2 * (2 - 2.5 * sin^2(i))
        double dw = common_factor * (2.0 - 2.5 * sin2_i) * dt;
        // dM/dt = n + (3/2) * n * J2 * (Re/p)^2 * sqrt(1-e^2) * (1.5 * cos^2(i) - 0.5)
        double dM_term = factor * ((3.0 * cos2_i - 1.0) / 2.0) * sqrt(one_minus_e2) / (n * a_val * a_val);
        double dM = (n + dM_term) * dt;
        
        // 更新轨道要素并将角度归一化到 [0, 2*PI) 范围。
        O[idx] = normalize_angle_cuda(O_val + dO);
        w[idx] = normalize_angle_cuda(w_val + dw);
        M[idx] = normalize_angle_cuda(M_val + dM);
    }
}

/**
 * @brief 将轨道要素转换为笛卡尔坐标的CUDA内核。
 *
 * 该内核为每个卫星启动一个线程，并行地将其轨道要素转换为地心惯性系(ECI)中的位置坐标(x, y, z)。
 *
 * @param a 指向半长轴数组的设备指针。
 * @param e 指向偏心率数组的设备指针。
 * @param i 指向倾角数组的设备指针。
 * @param O 指向升交点赤经数组的设备指针。
 * @param w 指向近地点幅角数组的设备指针。
 * @param M 指向平近点角数组的设备指针。
 * @param pos_x 指向x坐标数组的设备指针（输出）。
 * @param pos_y 指向y坐标数组的设备指针（输出）。
 * @param pos_z 指向z坐标数组的设备指针（输出）。
 * @param num_satellites 卫星总数。
 */
__global__ void compute_positions_kernel(double* a, double* e, double* i,
                                        double* O, double* w, double* M,
                                        double* pos_x, double* pos_y, double* pos_z,
                                        int num_satellites) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_satellites) {
        // 从全局内存加载轨道要素。
        double a_val = a[idx];
        double e_val = e[idx];
        double i_val = i[idx];
        double O_val = O[idx];
        double w_val = w[idx];
        double M_val = M[idx];
        
        // 求解开普勒方程 M = E - e*sin(E) 来计算偏近点角(E)。
        // 这里使用牛顿法进行一次迭代，对于小偏心率轨道足够精确。
        double E = M_val;  // 初始猜测
        E = E - (E - e_val * sin(E) - M_val) / (1.0 - e_val * cos(E));
        
        // 计算真近点角(nu)。
        double tan_nu_2 = sqrt((1.0 + e_val) / (1.0 - e_val)) * tan(E / 2.0);
        double nu = 2.0 * atan(tan_nu_2);
        
        // 计算卫星到地心的距离(r)。
        double r = a_val * (1.0 - e_val * cos(E));
        
        // 在轨道平面（周航坐标系）内计算位置。
        double x_perifocal = r * cos(nu);
        double y_perifocal = r * sin(nu);
        
        // 预计算从周航坐标系到地心惯性系的旋转矩阵所需的三角函数。
        double cosO = cos(O_val);
        double sinO = sin(O_val);
        double cosi = cos(i_val);
        double sini = sin(i_val);
        double cosw = cos(w_val);
        double sinw = sin(w_val);
        
        // 执行坐标旋转，将位置从周航坐标系转换到地心惯性系(ECI)。
        pos_x[idx] = (cosO*cosw - sinO*sinw*cosi) * x_perifocal + 
                     (-cosO*sinw - sinO*cosw*cosi) * y_perifocal;
        pos_y[idx] = (sinO*cosw + cosO*sinw*cosi) * x_perifocal + 
                     (-sinO*sinw + cosO*cosw*cosi) * y_perifocal;
        pos_z[idx] = sinw*sini * x_perifocal + cosw*sini * y_perifocal;
    }
}

// C接口函数，封装CUDA内核调用，以便从C++代码中调用。
extern "C" {
    /**
     * @brief J2轨道外推的CUDA接口函数。
     *
     * 该函数负责管理内存传输（主机到设备，设备到主机）和启动`j2_propagate_kernel`内核。
     *
     * @param elements 指向主机内存中轨道要素数据数组的指针（SoA布局）。
     * @param num_satellites 卫星数量。
     * @param dt 时间步长（秒）。
     * @param mu 地球引力常数。
     * @param re 地球赤道半径。
     * @param j2 地球J2摄动系数。
     */
    void cuda_propagate_j2(double* elements, size_t num_satellites, double dt, 
                          double mu, double re, double j2) {
        
        // 将物理常数从主机内存复制到设备的__constant__内存。
        cudaMemcpyToSymbol(d_MU, &mu, sizeof(double));
        cudaMemcpyToSymbol(d_RE, &re, sizeof(double));
        cudaMemcpyToSymbol(d_J2, &j2, sizeof(double));
        
        // 在GPU设备上为每个轨道要素数组分配内存。
        double *d_a, *d_e, *d_i, *d_O, *d_w, *d_M;
        size_t size = num_satellites * sizeof(double);
        
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_e, size);
        cudaMalloc(&d_i, size);
        cudaMalloc(&d_O, size);
        cudaMalloc(&d_w, size);
        cudaMalloc(&d_M, size);
        
        // 将轨道要素数据从主机内存(elements)复制到设备内存。
        // 数据按SoA布局，因此需要根据偏移量分别复制每个要素。
        cudaMemcpy(d_a, elements, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_e, elements + num_satellites, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_i, elements + 2*num_satellites, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_O, elements + 3*num_satellites, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, elements + 4*num_satellites, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_M, elements + 5*num_satellites, size, cudaMemcpyHostToDevice);
        
        // 计算CUDA内核启动配置：线程块大小和网格大小。
        int block_size = 256; // 每个线程块包含256个线程，这是一个常见的选择。
        int grid_size = (num_satellites + block_size - 1) / block_size; // 确保有足够的线程块来覆盖所有卫星。
        
        // 启动J2外推内核。
        j2_propagate_kernel<<<grid_size, block_size>>>(
            d_a, d_e, d_i, d_O, d_w, d_M, num_satellites, dt);
        
        // 检查内核启动或执行过程中是否发生错误。
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA kernel error in j2_propagate_kernel: %s\n", cudaGetErrorString(error));
        }
        
        // 将更新后的轨道要素（O, w, M）从设备内存复制回主机内存。
        cudaMemcpy(elements + 3*num_satellites, d_O, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(elements + 4*num_satellites, d_w, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(elements + 5*num_satellites, d_M, size, cudaMemcpyDeviceToHost);
        
        // 释放设备上分配的内存。
        cudaFree(d_a);
        cudaFree(d_e);
        cudaFree(d_i);
        cudaFree(d_O);
        cudaFree(d_w);
        cudaFree(d_M);
    }
    
    /**
     * @brief 从轨道要素计算位置的CUDA接口函数。
     *
     * 该函数负责管理内存传输和启动`compute_positions_kernel`内核。
     *
     * @param elements 指向主机内存中轨道要素数据数组的指针（SoA布局）。
     * @param positions 指向主机内存中用于存储位置坐标的数组指针（SoA布局）。
     * @param num_satellites 卫星数量。
     */
    void cuda_compute_positions(double* elements, double* positions, 
                               size_t num_satellites) {
        // 在GPU设备上为轨道要素和位置坐标分配内存。
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
        
        // 将轨道要素数据从主机内存复制到设备内存。
        cudaMemcpy(d_a, elements, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_e, elements + num_satellites, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_i, elements + 2*num_satellites, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_O, elements + 3*num_satellites, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, elements + 4*num_satellites, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_M, elements + 5*num_satellites, size, cudaMemcpyHostToDevice);
        
        // 计算CUDA内核启动配置。
        int block_size = 256;
        int grid_size = (num_satellites + block_size - 1) / block_size;
        
        // 启动位置计算内核。
        compute_positions_kernel<<<grid_size, block_size>>>(
            d_a, d_e, d_i, d_O, d_w, d_M, d_pos_x, d_pos_y, d_pos_z, num_satellites);

        // 检查内核启动或执行过程中是否发生错误。
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA kernel error in compute_positions_kernel: %s\n", cudaGetErrorString(error));
        }
        
        // 将计算出的位置坐标从设备内存复制回主机内存。
        cudaMemcpy(positions, d_pos_x, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(positions + num_satellites, d_pos_y, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(positions + 2*num_satellites, d_pos_z, size, cudaMemcpyDeviceToHost);
        
        // 释放设备上分配的所有内存。
        cudaFree(d_a); cudaFree(d_e); cudaFree(d_i);
        cudaFree(d_O); cudaFree(d_w); cudaFree(d_M);
        cudaFree(d_pos_x); cudaFree(d_pos_y); cudaFree(d_pos_z);
    }
}

#endif // __CUDACC__