#version 450

layout(location = 0) in vec3 fragPos;       // 世界空间位置
layout(location = 1) in vec3 fragNormal;    // 世界空间法向量
layout(location = 2) in vec2 fragTexCoord;  // 纹理坐标
layout(location = 3) in vec3 fragLightPos;  // 光源位置
layout(location = 4) in vec3 fragViewPos;   // 观察者位置

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D earthTexture;    // 地球表面纹理
layout(binding = 2) uniform sampler2D cloudTexture;    // 云层纹理

void main() {
    // 光照计算
    vec3 lightDir = normalize(fragLightPos - fragPos);
    vec3 viewDir = normalize(fragViewPos - fragPos);
    vec3 normal = normalize(fragNormal);
    
    // 环境光
    float ambient = 0.3;
    
    // 漫反射
    float diff = max(dot(normal, lightDir), 0.0);
    
    // 镜面反射
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    
    // 采样地球表面纹理
    vec4 earthColor = texture(earthTexture, fragTexCoord);
    
    // 采样云层纹理
    vec4 cloudColor = texture(cloudTexture, fragTexCoord);
    
    // 如果纹理采样失败，使用默认颜色
    if (earthColor.a < 0.1) {
        earthColor = vec4(0.2, 0.5, 0.8, 1.0);  // 默认蓝色地球
    }
    if (cloudColor.a < 0.1) {
        cloudColor = vec4(1.0, 1.0, 1.0, 0.0);  // 默认透明云层
    }
    
    // 地球表面光照计算
    vec3 earthLit = earthColor.rgb * (ambient + diff * 0.7 + spec * 0.3);
    
    // 云层光照计算（云层受光照影响较小，无镜面反射）
    vec3 cloudLit = cloudColor.rgb * (0.8 + diff * 0.2);
    
    // 混合地球表面和云层
    // 使用云层的alpha值作为混合因子
    float cloudAlpha = cloudColor.a * 0.8;  // 调整云层透明度
    vec3 finalColor = mix(earthLit, cloudLit, cloudAlpha);
    
    outColor = vec4(finalColor, 1.0);
}