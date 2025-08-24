#version 450

/**
 * @brief 轨道渲染顶点着色器
 * 处理轨道路径和卫星的顶点变换
 */

// 顶点属性
layout(location = 0) in vec3 inPosition;    // 顶点位置
layout(location = 1) in vec3 inColor;       // 顶点颜色
layout(location = 2) in float inTimestamp;  // 时间戳（用于动画）

// 统一缓冲区对象
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;         // 模型矩阵
    mat4 view;          // 视图矩阵
    mat4 proj;          // 投影矩阵
    vec3 cameraPos;     // 相机位置
    float currentTime;  // 当前时间
    float orbitAlpha;   // 轨道透明度
    float pointSize;    // 点大小
} ubo;

// 输出到片段着色器
layout(location = 0) out vec3 fragColor;     // 颜色
layout(location = 1) out float fragAlpha;    // 透明度
layout(location = 2) out vec3 fragPos;       // 世界空间位置
layout(location = 3) out float fragTime;     // 时间因子

void main() {
    // 计算世界空间位置
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    fragPos = worldPos.xyz;
    
    // 传递颜色
    fragColor = inColor;
    
    // 计算时间相关的透明度
    float timeFactor = 1.0;
    if (ubo.currentTime > 0.0 && inTimestamp > 0.0) {
        // 根据时间差计算淡出效果
        float timeDiff = abs(ubo.currentTime - inTimestamp);
        timeFactor = exp(-timeDiff * 0.001); // 指数衰减
    }
    
    fragAlpha = ubo.orbitAlpha * timeFactor;
    fragTime = timeFactor;
    
    // 计算距离相关的点大小
    float distance = length(ubo.cameraPos - worldPos.xyz);
    float sizeScale = 1.0 / (1.0 + distance * 0.0001);
    gl_PointSize = ubo.pointSize * sizeScale;
    
    // 计算最终的顶点位置
    gl_Position = ubo.proj * ubo.view * worldPos;
}