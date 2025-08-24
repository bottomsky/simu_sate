#pragma once

#include "visualization_types.h"
#include "vulkan_renderer.h"
#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include <unordered_map>

namespace j2_orbit_visualization {

/**
 * @brief 轨道数据结构
 * 存储单个轨道的完整信息
 */
struct OrbitData {
    std::vector<OrbitPoint> points;       ///< 轨道点集合
    glm::vec3 color;                      ///< 轨道颜色
    float lineWidth;                      ///< 线宽
    bool visible;                         ///< 是否可见
    std::string name;                     ///< 轨道名称
    uint32_t satelliteId;                 ///< 卫星ID
    uint32_t id;                          ///< 轨道ID
};

/**
 * @brief 卫星渲染数据
 * 存储卫星的渲染信息
 */
struct SatelliteRenderData {
    SatelliteState state;                 ///< 卫星状态
    glm::vec3 color;                      ///< 卫星颜色
    float scale;                          ///< 卫星缩放
    bool visible;                         ///< 是否可见
    std::string name;                     ///< 卫星名称
    uint32_t id;                          ///< 卫星ID
};

/**
 * @brief 轨道渲染器类
 * 负责渲染卫星轨道路径和卫星位置
 */
class OrbitRenderer {
public:
    /**
     * @brief 构造函数
     * @param renderer Vulkan 渲染器引用
     */
    explicit OrbitRenderer(VulkanRenderer& renderer);
    
    /**
     * @brief 析构函数
     * 清理轨道渲染相关资源
     */
    ~OrbitRenderer();
    
    /**
     * @brief 初始化轨道渲染器
     * @return VisualizationError 初始化结果
     */
    VisualizationError initialize();
    
    /**
     * @brief 渲染所有轨道和卫星
     * @param commandBuffer 命令缓冲区
     * @param camera 相机参数
     * @return VisualizationError 渲染结果
     */
    VisualizationError render(VkCommandBuffer commandBuffer, const CameraParams& camera);
    
    /**
     * @brief 添加轨道数据
     * @param points 轨道点集合
     * @param color 轨道颜色
     * @param visible 是否可见
     * @return uint32_t 轨道ID
     */
    uint32_t addOrbit(const std::vector<OrbitPoint>& points, const glm::vec3& color, bool visible);
    
    /**
     * @brief 更新轨道数据
     * @param orbitId 轨道ID
     * @param points 新的轨道点集合
     */
    void updateOrbit(uint32_t orbitId, const std::vector<OrbitPoint>& points);
    
    /**
     * @brief 移除轨道
     * @param orbitId 轨道ID
     */
    void removeOrbit(uint32_t orbitId);
    
    /**
     * @brief 添加卫星
     * @param state 卫星状态
     * @param color 卫星颜色
     * @param scale 卫星缩放
     * @param visible 是否可见
     * @return uint32_t 卫星ID
     */
    uint32_t addSatellite(const SatelliteState& state, const glm::vec3& color, float scale, bool visible);
    
    /**
     * @brief 更新卫星状态
     * @param satelliteId 卫星ID
     * @param state 新的卫星状态
     */
    void updateSatellite(uint32_t satelliteId, const SatelliteState& state);
    
    /**
     * @brief 移除卫星
     * @param satelliteId 卫星ID
     */
    void removeSatellite(uint32_t satelliteId);
    
    /**
     * @brief 设置轨道可见性
     * @param orbitId 轨道ID
     * @param visible 是否可见
     */
    void setOrbitVisible(uint32_t orbitId, bool visible);
    
    /**
     * @brief 设置卫星可见性
     * @param satelliteId 卫星ID
     * @param visible 是否可见
     */
    void setSatelliteVisible(uint32_t satelliteId, bool visible);
    
    /**
     * @brief 设置轨道颜色
     * @param orbitId 轨道ID
     * @param color 新颜色
     */
    void setOrbitColor(uint32_t orbitId, const glm::vec3& color);
    
    /**
     * @brief 设置卫星颜色
     * @param satelliteId 卫星ID
     * @param color 新颜色
     */
    void setSatelliteColor(uint32_t satelliteId, const glm::vec3& color);
    
    /**
     * @brief 清除所有轨道
     */
    void clearOrbits();
    
    /**
     * @brief 清除所有卫星
     */
    void clearSatellites();
    
    /**
     * @brief 清除所有轨道和卫星
     */
    void clear();
    
    /**
     * @brief 获取轨道数量
     * @return size_t 轨道数量
     */
    size_t getOrbitCount() const { return orbits.size(); }
    
    /**
     * @brief 获取卫星数量
     * @return size_t 卫星数量
     */
    size_t getSatelliteCount() const { return satellites.size(); }
    
    /**
     * @brief 设置轨道线宽
     * @param lineWidth 线宽
     */
    void setDefaultLineWidth(float lineWidth) { defaultLineWidth = lineWidth; }
    
    /**
     * @brief 设置卫星大小
     * @param size 卫星大小
     */
    void setDefaultSatelliteSize(float size) { defaultSatelliteSize = size; }

private:
    VulkanRenderer& vulkanRenderer;       ///< Vulkan 渲染器引用
    
    // 轨道与卫星容器（ID 索引）
    std::unordered_map<uint32_t, OrbitData> orbits;                  ///< 轨道数据映射
    std::unordered_map<uint32_t, SatelliteRenderData> satellites;    ///< 卫星数据映射
    
    // 轨道线渲染资源
    VulkanBuffer orbitVertexBuffer;       ///< 轨道顶点缓冲区
    uint32_t orbitVertexCount = 0;        ///< 轨道顶点数量
    
    // 卫星渲染资源
    VulkanBuffer satelliteVertexBuffer;   ///< 卫星顶点缓冲区
    VulkanBuffer satelliteIndexBuffer;    ///< 卫星索引缓冲区
    uint32_t satelliteIndexCount = 0;     ///< 卫星索引数量
    
    // 管线与描述符
    VkDescriptorSetLayout descriptorSetLayout;   ///< 通用描述符集布局
    VkPipelineLayout orbitPipelineLayout;        ///< 轨道管线布局
    VkPipeline orbitPipeline;                    ///< 轨道图形管线
    VkPipelineLayout satellitePipelineLayout;    ///< 卫星管线布局
    VkPipeline satellitePipeline;                ///< 卫星图形管线
    VkDescriptorPool descriptorPool;             ///< 描述符池
    std::vector<VkDescriptorSet> descriptorSets; ///< 描述符集（按帧）
    
    // 统一缓冲区
    std::vector<VulkanBuffer> uniformBuffers;    ///< 通用统一缓冲区（按帧）
    
    // 渲染参数
    float defaultLineWidth = 2.0f;       ///< 默认线宽
    float defaultSatelliteSize = 5.0f;   ///< 默认卫星大小

    // 自增 ID 计数器
    uint32_t nextOrbitId = 1;            ///< 下一个可用的轨道ID
    uint32_t nextSatelliteId = 1;        ///< 下一个可用的卫星ID
    
    /**
     * @brief 创建描述符集布局
     */
    VisualizationError createDescriptorSetLayout();
    
    /**
     * @brief 创建轨道渲染管线
     */
    VisualizationError createOrbitPipeline();
    
    /**
     * @brief 创建卫星渲染管线
     */
    VisualizationError createSatellitePipeline();
    
    /**
     * @brief 创建统一缓冲区
     */
    VisualizationError createUniformBuffers();
    
    /**
     * @brief 创建描述符池
     */
    VisualizationError createDescriptorPool();
    
    /**
     * @brief 创建描述符集
     */
    VisualizationError createDescriptorSets();
    
    /**
     * @brief 创建卫星几何数据
     */
    VisualizationError createSatelliteGeometry();
    
    /**
     * @brief 更新统一缓冲区
     */
    void updateUniformBuffer(uint32_t frameIndex, const CameraParams& camera);
    
    /**
     * @brief 更新轨道顶点缓冲区
     */
    void updateOrbitVertexBuffer();
    
    /**
     * @brief 渲染轨道
     */
    void renderOrbits(VkCommandBuffer commandBuffer, uint32_t frameIndex);
    
    /**
     * @brief 渲染卫星
     */
    void renderSatellites(VkCommandBuffer commandBuffer, uint32_t frameIndex);
    
    /**
     * @brief 清理资源
     */
    void cleanup();
};

} // namespace j2_orbit_visualization