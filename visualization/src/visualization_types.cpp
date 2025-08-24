#include "visualization_types.h"
#include <cstddef>  // for offsetof

/**
 * @brief 获取 Vulkan 顶点绑定描述
 * @return VkVertexInputBindingDescription 顶点绑定描述
 */
VkVertexInputBindingDescription j2_orbit_visualization::OrbitPoint::getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(j2_orbit_visualization::OrbitPoint);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
}

/**
 * @brief 获取 Vulkan 顶点属性描述
 * @return std::vector<VkVertexInputAttributeDescription> 属性描述数组
 */
std::vector<VkVertexInputAttributeDescription> j2_orbit_visualization::OrbitPoint::getAttributeDescriptions() {
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions(3);

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(j2_orbit_visualization::OrbitPoint, position);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(j2_orbit_visualization::OrbitPoint, color);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(j2_orbit_visualization::OrbitPoint, timestamp);

    return attributeDescriptions;
}

/**
 * @brief 获取 Vulkan 顶点绑定描述 (Vertex)
 * @return VkVertexInputBindingDescription 顶点绑定描述
 */
VkVertexInputBindingDescription j2_orbit_visualization::Vertex::getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(j2_orbit_visualization::Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
}

/**
 * @brief 获取 Vulkan 顶点属性描述 (Vertex)
 * @return std::vector<VkVertexInputAttributeDescription> 属性描述数组
 */
std::vector<VkVertexInputAttributeDescription> j2_orbit_visualization::Vertex::getAttributeDescriptions() {
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions(3);

    // 位置属性
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(j2_orbit_visualization::Vertex, position);

    // 法线属性
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(j2_orbit_visualization::Vertex, normal);

    // 纹理坐标属性
    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(j2_orbit_visualization::Vertex, texCoord);

    return attributeDescriptions;
}