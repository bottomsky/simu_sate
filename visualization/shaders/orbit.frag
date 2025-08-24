#version 450

/**
 * @brief 轨道渲染片段着色器
 * 实现轨道路径和卫星的渲染效果
 */

// 从顶点着色器输入
layout(location = 0) in vec3 fragColor;     // 颜色
layout(location = 1) in float fragAlpha;    // 透明度
layout(location = 2) in vec3 fragPos;       // 世界空间位置
layout(location = 3) in float fragTime;     // 时间因子

// 推送常量
layout(push_constant) uniform PushConstants {
    int renderMode;        // 渲染模式：0=轨道线，1=轨道点，2=卫星
    float glowIntensity;   // 发光强度
    float fadeDistance;    // 淡出距离
    vec3 highlightColor;   // 高亮颜色
} pushConstants;

// 输出颜色
layout(location = 0) out vec4 outColor;

/**
 * @brief 计算轨道线的渲染效果
 * @return vec4 最终颜色
 */
vec4 renderOrbitLine() {
    vec3 color = fragColor;
    float alpha = fragAlpha;
    
    // 添加发光效果
    if (pushConstants.glowIntensity > 0.0) {
        color += pushConstants.highlightColor * pushConstants.glowIntensity;
    }
    
    // 时间淡出效果
    alpha *= fragTime;
    
    return vec4(color, alpha);
}

/**
 * @brief 计算轨道点的渲染效果
 * @return vec4 最终颜色
 */
vec4 renderOrbitPoint() {
    // 计算点的圆形形状
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    // 创建圆形点
    if (dist > 0.5) {
        discard;
    }
    
    vec3 color = fragColor;
    float alpha = fragAlpha;
    
    // 添加径向渐变效果
    float radialFade = 1.0 - (dist * 2.0);
    alpha *= radialFade;
    
    // 添加发光效果
    if (pushConstants.glowIntensity > 0.0) {
        color += pushConstants.highlightColor * pushConstants.glowIntensity * radialFade;
    }
    
    // 时间淡出效果
    alpha *= fragTime;
    
    return vec4(color, alpha);
}

/**
 * @brief 计算卫星的渲染效果
 * @return vec4 最终颜色
 */
vec4 renderSatellite() {
    // 计算点的圆形形状
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    // 创建圆形卫星
    if (dist > 0.5) {
        discard;
    }
    
    vec3 color = fragColor;
    float alpha = fragAlpha;
    
    // 卫星的特殊效果
    // 中心亮点
    float centerGlow = 1.0 - smoothstep(0.0, 0.2, dist);
    color += vec3(1.0) * centerGlow * 0.5;
    
    // 外围发光
    float outerGlow = 1.0 - smoothstep(0.3, 0.5, dist);
    color += pushConstants.highlightColor * outerGlow * pushConstants.glowIntensity;
    
    // 脉动效果（基于时间）
    float pulse = sin(fragTime * 10.0) * 0.1 + 0.9;
    alpha *= pulse;
    
    return vec4(color, alpha);
}

void main() {
    vec4 finalColor;
    
    // 根据渲染模式选择不同的渲染方法
    switch (pushConstants.renderMode) {
        case 0: // 轨道线
            finalColor = renderOrbitLine();
            break;
        case 1: // 轨道点
            finalColor = renderOrbitPoint();
            break;
        case 2: // 卫星
            finalColor = renderSatellite();
            break;
        default:
            finalColor = vec4(fragColor, fragAlpha);
            break;
    }
    
    // 应用距离淡出
    if (pushConstants.fadeDistance > 0.0) {
        float distance = length(fragPos);
        float fadeFactor = 1.0 - smoothstep(pushConstants.fadeDistance * 0.8, pushConstants.fadeDistance, distance);
        finalColor.a *= fadeFactor;
    }
    
    // 确保透明度在合理范围内
    finalColor.a = clamp(finalColor.a, 0.0, 1.0);
    
    // 如果透明度太低，丢弃片段
    if (finalColor.a < 0.01) {
        discard;
    }
    
    outColor = finalColor;
}