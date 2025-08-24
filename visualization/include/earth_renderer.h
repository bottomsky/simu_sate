#pragma once

#include "visualization_types.h"
#include "vulkan_renderer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>

namespace j2_orbit_visualization {

/**
 * @brief 地球渲染器类
 * 负责渲染地球球体，包括纹理映射和光照效果
 */
class EarthRenderer {
public:
    /**
     * @brief 构造函数
     * @param renderer Vulkan 渲染器引用
     */
    explicit EarthRenderer(VulkanRenderer& renderer);
    
    /**
     * @brief 析构函数
     * 清理地球渲染相关资源
     */
    ~EarthRenderer();
    
    /**
     * @brief 初始化地球渲染器
     * @param textureFile 地球纹理文件路径
     * @param cloudTextureFile 云层纹理文件路径（可选）
     * @param normalMapFile 法线贴图文件路径（可选）
     * @return VisualizationError 初始化结果
     */
    VisualizationError initialize(const std::string& textureFile = "", 
                                 const std::string& cloudTextureFile = "",
                                 const std::string& normalMapFile = "");
    
    /**
     * @brief 渲染地球
     * @param commandBuffer 命令缓冲区
     * @param renderParams 地球渲染参数
     * @param camera 相机参数
     * @return VisualizationError 渲染结果
     */
    VisualizationError render(VkCommandBuffer commandBuffer, 
                             const EarthRenderParams& renderParams,
                             const CameraParams& camera);
    
    /**
     * @brief 更新地球渲染参数
     * @param params 新的渲染参数
     */
    void updateRenderParams(const EarthRenderParams& params);
    
    /**
     * @brief 设置地球半径
     * @param radius 地球半径（单位：千米）
     */
    void setEarthRadius(float radius) { earthRadius = radius; }
    
    /**
     * @brief 获取地球半径
     * @return float 地球半径
     */
    float getEarthRadius() const { return earthRadius; }
    
    /**
     * @brief 设置地球自转角度
     * @param angle 自转角度（弧度）
     */
    void setRotationAngle(float angle) { rotationAngle = angle; }
    
    /**
     * @brief 获取地球自转角度
     * @return float 自转角度
     */
    float getRotationAngle() const { return rotationAngle; }
    
    /**
     * @brief 启用/禁用大气效果
     * @param enable 是否启用
     */
    void setAtmosphereEnabled(bool enable) { atmosphereEnabled = enable; }
    
    /**
     * @brief 检查大气效果是否启用
     * @return bool 是否启用大气效果
     */
    bool isAtmosphereEnabled() const { return atmosphereEnabled; }

private:
    VulkanRenderer& vulkanRenderer;       ///< Vulkan 渲染器引用
    
    // 几何数据
    VulkanBuffer vertexBuffer;            ///< 顶点缓冲区
    VulkanBuffer indexBuffer;             ///< 索引缓冲区
    uint32_t indexCount;                  ///< 索引数量
    
    // 纹理资源
    VulkanImage earthTexture;             ///< 地球表面纹理
    VulkanImage cloudTexture;             ///< 云层纹理
    VulkanImage normalMap;                ///< 法线贴图
    VkSampler textureSampler;             ///< 纹理采样器
    
    // 渲染管线
    VkDescriptorSetLayout descriptorSetLayout; ///< 描述符集布局
    VkPipelineLayout pipelineLayout;      ///< 管线布局
    VkPipeline graphicsPipeline;          ///< 图形管线
    
    // 描述符相关
    VkDescriptorPool descriptorPool;      ///< 描述符池
    std::vector<VkDescriptorSet> descriptorSets; ///< 描述符集
    
    // 统一缓冲区
    std::vector<VulkanBuffer> uniformBuffers; ///< 统一缓冲区
    
    // 地球参数
    float earthRadius = 6371.0f;         ///< 地球半径（千米）
    float rotationAngle = 0.0f;          ///< 地球自转角度
    bool atmosphereEnabled = true;        ///< 是否启用大气效果
    
    // 球体细分参数
    uint32_t latitudeSegments = 32;       ///< 纬度分段数
    uint32_t longitudeSegments = 32;      ///< 经度分段数
    
    /**
     * @brief 生成球体几何数据
     * @param vertices 输出顶点数据
     * @param indices 输出索引数据
     */
    void generateSphereGeometry(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices);
    
    /**
     * @brief 创建顶点缓冲区
     * @param vertices 顶点数据
     * @return VisualizationError 创建结果
     */
    VisualizationError createVertexBuffer(const std::vector<Vertex>& vertices);
    
    /**
     * @brief 创建索引缓冲区
     * @param indices 索引数据
     * @return VisualizationError 创建结果
     */
    VisualizationError createIndexBuffer(const std::vector<uint32_t>& indices);
    
    /**
     * @brief 创建纹理图像
     * @param textureFile 纹理文件路径
     * @param image 输出图像结构
     * @return VisualizationError 创建结果
     */
    VisualizationError createTextureImage(const std::string& textureFile, VulkanImage& image);
    
    /**
     * @brief 创建纹理采样器
     * @return VisualizationError 创建结果
     */
    VisualizationError createTextureSampler();
    
    /**
     * @brief 创建描述符集布局
     * @return VisualizationError 创建结果
     */
    VisualizationError createDescriptorSetLayout();
    
    /**
     * @brief 创建图形管线
     * @return VisualizationError 创建结果
     */
    VisualizationError createGraphicsPipeline();
    
    /**
     * @brief 创建统一缓冲区
     * @return VisualizationError 创建结果
     */
    VisualizationError createUniformBuffers();
    
    /**
     * @brief 创建描述符池
     * @return VisualizationError 创建结果
     */
    VisualizationError createDescriptorPool();
    
    /**
     * @brief 创建描述符集
     * @return VisualizationError 创建结果
     */
    VisualizationError createDescriptorSets();
    
    /**
     * @brief 更新统一缓冲区
     * @param frameIndex 当前帧索引
     * @param renderParams 渲染参数
     * @param camera 相机参数
     */
    void updateUniformBuffer(uint32_t frameIndex, const EarthRenderParams& renderParams, 
                           const CameraParams& camera);
    
    /**
     * @brief 加载着色器模块
     * @param filename 着色器文件名
     * @return VkShaderModule 着色器模块
     */
    VkShaderModule loadShaderModule(const std::string& filename);
    
    /**
     * @brief 清理资源
     */
    void cleanup();
};

} // namespace j2_orbit_visualization