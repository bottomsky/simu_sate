#version 450

/**
 * @brief 地球渲染顶点着色器
 * 处理地球球体的顶点变换和纹理坐标计算
 */

// 顶点属性
layout(location = 0) in vec3 inPosition;    // 顶点位置
layout(location = 1) in vec3 inNormal;      // 顶点法向量
layout(location = 2) in vec2 inTexCoord;    // 纹理坐标

// 统一缓冲区对象
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;         // 模型矩阵
    mat4 view;          // 视图矩阵
    mat4 proj;          // 投影矩阵
    mat4 normalMatrix;  // 法向量矩阵
    vec3 lightPos;      // 光源位置
    vec3 viewPos;       // 观察者位置
    float time;         // 时间（用于动画）
} ubo;

// 输出到片段着色器
layout(location = 0) out vec3 fragPos;      // 世界空间位置
layout(location = 1) out vec3 fragNormal;   // 世界空间法向量
layout(location = 2) out vec2 fragTexCoord; // 纹理坐标
layout(location = 3) out vec3 fragLightPos; // 光源位置
layout(location = 4) out vec3 fragViewPos;  // 观察者位置

void main() {
    // 计算世界空间位置
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    fragPos = worldPos.xyz;
    
    // 计算世界空间法向量
    fragNormal = normalize(mat3(ubo.normalMatrix) * inNormal);
    
    // 传递纹理坐标
    fragTexCoord = inTexCoord;
    
    // 传递光源和观察者位置
    fragLightPos = ubo.lightPos;
    fragViewPos = ubo.viewPos;
    
    // 计算最终的顶点位置
    gl_Position = ubo.proj * ubo.view * worldPos;
}