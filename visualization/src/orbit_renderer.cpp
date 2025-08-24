#include "orbit_renderer.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace j2_orbit_visualization {

// 与 VulkanRenderer 保持一致的飞行中帧数常量
static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

/**
 * @brief 构造函数
 * @param renderer Vulkan 渲染器引用
 */
OrbitRenderer::OrbitRenderer(VulkanRenderer& renderer)
    : vulkanRenderer(renderer), descriptorSetLayout(VK_NULL_HANDLE),
      orbitPipelineLayout(VK_NULL_HANDLE), satellitePipelineLayout(VK_NULL_HANDLE),
      orbitPipeline(VK_NULL_HANDLE), satellitePipeline(VK_NULL_HANDLE),
      descriptorPool(VK_NULL_HANDLE), nextOrbitId(1), nextSatelliteId(1) {
    
    // 初始化缓冲区结构
    orbitVertexBuffer = {};
    satelliteVertexBuffer = {};
    satelliteIndexBuffer = {};
}

/**
 * @brief 析构函数
 * 清理轨道渲染相关资源
 */
OrbitRenderer::~OrbitRenderer() {
    cleanup();
}

/**
 * @brief 初始化轨道渲染器
 * @return VisualizationError 初始化结果
 */
VisualizationError OrbitRenderer::initialize() {
    // 创建描述符集布局
    auto result = createDescriptorSetLayout();
    if (result != VisualizationError::Success) {
        return result;
    }
    
    // 创建轨道渲染管线
    result = createOrbitPipeline();
    if (result != VisualizationError::Success) {
        return result;
    }
    
    // 创建卫星渲染管线
    result = createSatellitePipeline();
    if (result != VisualizationError::Success) {
        return result;
    }
    
    // 创建卫星几何数据
    result = createSatelliteGeometry();
    if (result != VisualizationError::Success) {
        return result;
    }
    
    // 创建统一缓冲区
    result = createUniformBuffers();
    if (result != VisualizationError::Success) {
        return result;
    }
    
    // 创建描述符池和描述符集
    result = createDescriptorPool();
    if (result != VisualizationError::Success) {
        return result;
    }
    
    result = createDescriptorSets();
    if (result != VisualizationError::Success) {
        return result;
    }
    
    return VisualizationError::Success;
}

/**
 * @brief 渲染轨道和卫星
 * @param commandBuffer 命令缓冲区
 * @param camera 相机参数
 * @return VisualizationError 渲染结果
 */
VisualizationError OrbitRenderer::render(VkCommandBuffer commandBuffer, const CameraParams& camera) {
    uint32_t frameIndex = vulkanRenderer.getCurrentFrameIndex();
    
    // 更新统一缓冲区
    updateUniformBuffer(frameIndex, camera);
    
    // 渲染轨道路径
    renderOrbits(commandBuffer, frameIndex);
    
    // 渲染卫星
    renderSatellites(commandBuffer, frameIndex);
    
    return VisualizationError::Success;
}

/**
 * @brief 添加轨道数据
 * @param points 轨道点数据
 * @param color 轨道颜色
 * @param visible 是否可见
 * @return uint32_t 轨道ID
 */
uint32_t OrbitRenderer::addOrbit(const std::vector<OrbitPoint>& points, 
                                const glm::vec3& color, bool visible) {
    uint32_t orbitId = nextOrbitId++;
    
    OrbitData orbitData;
    orbitData.id = orbitId;
    orbitData.points = points;
    orbitData.color = color;
    orbitData.visible = visible;
    orbitData.lineWidth = 1.0f;
    
    orbits[orbitId] = orbitData;
    
    // 更新轨道顶点缓冲区
    updateOrbitVertexBuffer();
    
    return orbitId;
}

/**
 * @brief 更新轨道数据
 * @param orbitId 轨道ID
 * @param points 新的轨道点数据
 */
void OrbitRenderer::updateOrbit(uint32_t orbitId, const std::vector<OrbitPoint>& points) {
    auto it = orbits.find(orbitId);
    if (it != orbits.end()) {
        it->second.points = points;
        updateOrbitVertexBuffer();
    }
}

/**
 * @brief 移除轨道
 * @param orbitId 轨道ID
 */
void OrbitRenderer::removeOrbit(uint32_t orbitId) {
    auto it = orbits.find(orbitId);
    if (it != orbits.end()) {
        orbits.erase(it);
        updateOrbitVertexBuffer();
    }
}

/**
 * @brief 设置轨道可见性
 * @param orbitId 轨道ID
 * @param visible 是否可见
 */
void OrbitRenderer::setOrbitVisible(uint32_t orbitId, bool visible) {
    auto it = orbits.find(orbitId);
    if (it != orbits.end()) {
        it->second.visible = visible;
    }
}

/**
 * @brief 设置轨道颜色
 * @param orbitId 轨道ID
 * @param color 新颜色
 */
void OrbitRenderer::setOrbitColor(uint32_t orbitId, const glm::vec3& color) {
    auto it = orbits.find(orbitId);
    if (it != orbits.end()) {
        it->second.color = color;
        updateOrbitVertexBuffer();
    }
}

/**
 * @brief 添加卫星
 * @param state 卫星状态
 * @param color 卫星颜色
 * @param scale 卫星缩放
 * @param visible 是否可见
 * @return uint32_t 卫星ID
 */
uint32_t OrbitRenderer::addSatellite(const SatelliteState& state, const glm::vec3& color, 
                                    float scale, bool visible) {
    uint32_t satelliteId = nextSatelliteId++;
    
    SatelliteRenderData satelliteData;
    satelliteData.id = satelliteId;
    satelliteData.state = state;
    satelliteData.color = color;
    satelliteData.scale = scale;
    satelliteData.visible = visible;
    
    satellites[satelliteId] = satelliteData;
    
    return satelliteId;
}

/**
 * @brief 更新卫星状态
 * @param satelliteId 卫星ID
 * @param state 新的卫星状态
 */
void OrbitRenderer::updateSatellite(uint32_t satelliteId, const SatelliteState& state) {
    auto it = satellites.find(satelliteId);
    if (it != satellites.end()) {
        it->second.state = state;
    }
}

/**
 * @brief 移除卫星
 * @param satelliteId 卫星ID
 */
void OrbitRenderer::removeSatellite(uint32_t satelliteId) {
    auto it = satellites.find(satelliteId);
    if (it != satellites.end()) {
        satellites.erase(it);
    }
}

/**
 * @brief 设置卫星可见性
 * @param satelliteId 卫星ID
 * @param visible 是否可见
 */
void OrbitRenderer::setSatelliteVisible(uint32_t satelliteId, bool visible) {
    auto it = satellites.find(satelliteId);
    if (it != satellites.end()) {
        it->second.visible = visible;
    }
}

/**
 * @brief 设置卫星颜色
 * @param satelliteId 卫星ID
 * @param color 新颜色
 */
void OrbitRenderer::setSatelliteColor(uint32_t satelliteId, const glm::vec3& color) {
    auto it = satellites.find(satelliteId);
    if (it != satellites.end()) {
        it->second.color = color;
    }
}

/**
 * @brief 清除所有轨道和卫星
 */
void OrbitRenderer::clear() {
    orbits.clear();
    satellites.clear();
    updateOrbitVertexBuffer();
}

/**
 * @brief 创建描述符集布局
 * @return VisualizationError 创建结果
 */
VisualizationError OrbitRenderer::createDescriptorSetLayout() {
    // 统一缓冲区绑定
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;
    
    if (vkCreateDescriptorSetLayout(vulkanRenderer.getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        return VisualizationError::VulkanError;
    }
    
    return VisualizationError::Success;
}

/**
 * @brief 创建轨道渲染管线
 * @return VisualizationError 创建结果
 */
VisualizationError OrbitRenderer::createOrbitPipeline() {
    auto vertShaderCode = vulkanRenderer.readFile(std::string(SHADER_DIR) + "/orbit.vert.spv");
    auto fragShaderCode = vulkanRenderer.readFile(std::string(SHADER_DIR) + "/orbit.frag.spv");

    if (vertShaderCode.empty() || fragShaderCode.empty()) {
        return VisualizationError::SHADER_COMPILATION_FAILED;
    }

    VkShaderModule vertShaderModule = vulkanRenderer.createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = vulkanRenderer.createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    auto bindingDescription = OrbitPoint::getBindingDescription();
    auto attributeDescriptions = OrbitPoint::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_LINE_WIDTH
    };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(vulkanRenderer.getDevice(), &pipelineLayoutInfo, nullptr, &orbitPipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanRenderer.getDevice(), fragShaderModule, nullptr);
        vkDestroyShaderModule(vulkanRenderer.getDevice(), vertShaderModule, nullptr);
        return VisualizationError::VulkanError;
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = orbitPipelineLayout;
    pipelineInfo.renderPass = vulkanRenderer.getRenderPass();
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(vulkanRenderer.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &orbitPipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(vulkanRenderer.getDevice(), orbitPipelineLayout, nullptr);
        vkDestroyShaderModule(vulkanRenderer.getDevice(), fragShaderModule, nullptr);
        vkDestroyShaderModule(vulkanRenderer.getDevice(), vertShaderModule, nullptr);
        return VisualizationError::VulkanError;
    }

    vkDestroyShaderModule(vulkanRenderer.getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(vulkanRenderer.getDevice(), vertShaderModule, nullptr);

    return VisualizationError::Success;
}

/**
 * @brief 创建卫星渲染管线
 * @return VisualizationError 创建结果
 */
VisualizationError OrbitRenderer::createSatellitePipeline() {
    auto vertShaderCode = vulkanRenderer.readFile(std::string(SHADER_DIR) + "/satellite.vert.spv");
    auto fragShaderCode = vulkanRenderer.readFile(std::string(SHADER_DIR) + "/satellite.frag.spv");

    if (vertShaderCode.empty() || fragShaderCode.empty()) {
        return VisualizationError::SHADER_COMPILATION_FAILED;
    }

    VkShaderModule vertShaderModule = vulkanRenderer.createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = vulkanRenderer.createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(glm::mat4);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(vulkanRenderer.getDevice(), &pipelineLayoutInfo, nullptr, &satellitePipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanRenderer.getDevice(), fragShaderModule, nullptr);
        vkDestroyShaderModule(vulkanRenderer.getDevice(), vertShaderModule, nullptr);
        return VisualizationError::VulkanError;
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = satellitePipelineLayout;
    pipelineInfo.renderPass = vulkanRenderer.getRenderPass();
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(vulkanRenderer.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &satellitePipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(vulkanRenderer.getDevice(), satellitePipelineLayout, nullptr);
        vkDestroyShaderModule(vulkanRenderer.getDevice(), fragShaderModule, nullptr);
        vkDestroyShaderModule(vulkanRenderer.getDevice(), vertShaderModule, nullptr);
        return VisualizationError::VulkanError;
    }

    vkDestroyShaderModule(vulkanRenderer.getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(vulkanRenderer.getDevice(), vertShaderModule, nullptr);

    return VisualizationError::Success;
}

/**
 * @brief 创建卫星几何数据
 * @return VisualizationError 创建结果
 */
VisualizationError OrbitRenderer::createSatelliteGeometry() {
    // 创建球形卫星模型
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    
    // 球体参数
    const uint32_t latitudeSegments = 16;   // 纬度段数
    const uint32_t longitudeSegments = 32;  // 经度段数
    const float radius = 1.0f;              // 单位球体，实际大小通过变换矩阵控制
    
    // 生成球体顶点
    for (uint32_t lat = 0; lat <= latitudeSegments; ++lat) {
        float theta = lat * M_PI / latitudeSegments; // 0 到 π
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);
        
        for (uint32_t lon = 0; lon <= longitudeSegments; ++lon) {
            float phi = lon * 2.0f * M_PI / longitudeSegments; // 0 到 2π
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);
            
            Vertex vertex{};
            
            // 位置
            vertex.position.x = radius * sinTheta * cosPhi;
            vertex.position.y = radius * cosTheta;
            vertex.position.z = radius * sinTheta * sinPhi;
            
            // 法线（对于球体，法线就是归一化的位置向量）
            vertex.normal = glm::normalize(vertex.position);
            
            // 纹理坐标
            vertex.texCoord.x = (float)lon / longitudeSegments;
            vertex.texCoord.y = 1.0f - (float)lat / latitudeSegments;
            
            vertices.push_back(vertex);
        }
    }
    
    // 生成球体索引（逆时针顺序）
    for (uint32_t lat = 0; lat < latitudeSegments; ++lat) {
        for (uint32_t lon = 0; lon < longitudeSegments; ++lon) {
            uint32_t first = lat * (longitudeSegments + 1) + lon;
            uint32_t second = first + longitudeSegments + 1;
            
            // 第一个三角形（逆时针顺序）
            indices.push_back(first);
            indices.push_back(first + 1);
            indices.push_back(second);
            
            // 第二个三角形（逆时针顺序）
            indices.push_back(second);
            indices.push_back(first + 1);
            indices.push_back(second + 1);
        }
    }
    
    satelliteIndexCount = static_cast<uint32_t>(indices.size());
    
    // 创建顶点缓冲区
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    
    VulkanBuffer stagingBuffer;
    auto result = vulkanRenderer.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                            stagingBuffer);
    if (result != VisualizationError::Success) {
        return result;
    }
    
    void* data;
    vkMapMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(vulkanRenderer.getDevice(), stagingBuffer.memory);
    
    result = vulkanRenderer.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, satelliteVertexBuffer);
    if (result != VisualizationError::Success) {
        vkDestroyBuffer(vulkanRenderer.getDevice(), stagingBuffer.buffer, nullptr);
        vkFreeMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, nullptr);
        return result;
    }
    
    vulkanRenderer.copyBuffer(stagingBuffer.buffer, satelliteVertexBuffer.buffer, bufferSize);
    
    vkDestroyBuffer(vulkanRenderer.getDevice(), stagingBuffer.buffer, nullptr);
    vkFreeMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, nullptr);
    
    // 创建索引缓冲区
    bufferSize = sizeof(indices[0]) * indices.size();
    
    result = vulkanRenderer.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       stagingBuffer);
    if (result != VisualizationError::Success) {
        return result;
    }
    
    vkMapMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t)bufferSize);
    vkUnmapMemory(vulkanRenderer.getDevice(), stagingBuffer.memory);
    
    result = vulkanRenderer.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, satelliteIndexBuffer);
    if (result != VisualizationError::Success) {
        vkDestroyBuffer(vulkanRenderer.getDevice(), stagingBuffer.buffer, nullptr);
        vkFreeMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, nullptr);
        return result;
    }
    
    vulkanRenderer.copyBuffer(stagingBuffer.buffer, satelliteIndexBuffer.buffer, bufferSize);
    
    vkDestroyBuffer(vulkanRenderer.getDevice(), stagingBuffer.buffer, nullptr);
    vkFreeMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, nullptr);
    
    return VisualizationError::Success;
}

/**
 * @brief 更新轨道顶点缓冲区
 */
void OrbitRenderer::updateOrbitVertexBuffer() {
    std::vector<OrbitPoint> vertices;
    
    // 收集所有可见轨道的顶点数据
    for (const auto& [orbitId, orbitData] : orbits) {
        if (!orbitData.visible || orbitData.points.empty()) {
            continue;
        }
        
        for (const auto& point : orbitData.points) {
            OrbitPoint orbitPoint{};
            orbitPoint.position = point.position;
            orbitPoint.color = orbitData.color;
            orbitPoint.timestamp = point.timestamp;
            vertices.push_back(orbitPoint);
        }
    }
    
    if (vertices.empty()) {
        if (orbitVertexBuffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(vulkanRenderer.getDevice(), orbitVertexBuffer.buffer, nullptr);
            vkFreeMemory(vulkanRenderer.getDevice(), orbitVertexBuffer.memory, nullptr);
            orbitVertexBuffer = {};
            orbitVertexCount = 0;
        }
        return;
    }
    
    // 清理旧的轨道顶点缓冲区
    if (orbitVertexBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(vulkanRenderer.getDevice(), orbitVertexBuffer.buffer, nullptr);
        vkFreeMemory(vulkanRenderer.getDevice(), orbitVertexBuffer.memory, nullptr);
        orbitVertexBuffer = {};
    }
    
    // 创建新的轨道顶点缓冲区
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    
    VulkanBuffer stagingBuffer;
    auto result = vulkanRenderer.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                            stagingBuffer);
    if (result != VisualizationError::Success) {
        return;
    }
    
    void* data;
    vkMapMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(vulkanRenderer.getDevice(), stagingBuffer.memory);
    
    result = vulkanRenderer.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, orbitVertexBuffer);
    if (result != VisualizationError::Success) {
        vkDestroyBuffer(vulkanRenderer.getDevice(), stagingBuffer.buffer, nullptr);
        vkFreeMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, nullptr);
        return;
    }
    
    vulkanRenderer.copyBuffer(stagingBuffer.buffer, orbitVertexBuffer.buffer, bufferSize);
    
    vkDestroyBuffer(vulkanRenderer.getDevice(), stagingBuffer.buffer, nullptr);
    vkFreeMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, nullptr);
    
    orbitVertexCount = static_cast<uint32_t>(vertices.size());
}

/**
 * @brief 渲染轨道路径
 * @param commandBuffer 命令缓冲区
 * @param frameIndex 当前帧索引
 */
void OrbitRenderer::renderOrbits(VkCommandBuffer commandBuffer, uint32_t frameIndex) {
    if (orbitPipeline == VK_NULL_HANDLE || orbitVertexBuffer.buffer == VK_NULL_HANDLE || orbitVertexCount == 0) {
        return;
    }
    
    // 绑定轨道渲染管线
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, orbitPipeline);
    
    // 绑定描述符集
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           orbitPipelineLayout, 0, 1, &descriptorSets[frameIndex], 0, nullptr);
    
    // 绑定顶点缓冲区
    VkBuffer vertexBuffers[] = {orbitVertexBuffer.buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    
    // 绘制轨道线
    uint32_t vertexOffset = 0;
    for (const auto& [orbitId, orbitData] : orbits) {
        if (!orbitData.visible || orbitData.points.empty()) {
            continue;
        }
        
        uint32_t pointCount = static_cast<uint32_t>(orbitData.points.size());
        if (pointCount > 1) {
            vkCmdDraw(commandBuffer, pointCount, 1, vertexOffset, 0);
        }
        vertexOffset += pointCount;
    }
}

/**
 * @brief 渲染卫星
 * @param commandBuffer 命令缓冲区
 * @param frameIndex 当前帧索引
 */
void OrbitRenderer::renderSatellites(VkCommandBuffer commandBuffer, uint32_t frameIndex) {
    if (satellitePipeline == VK_NULL_HANDLE || satelliteVertexBuffer.buffer == VK_NULL_HANDLE) {
        return;
    }
    
    // 绑定卫星渲染管线
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, satellitePipeline);
    
    // 绑定描述符集
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           satellitePipelineLayout, 0, 1, &descriptorSets[frameIndex], 0, nullptr);
    
    // 绑定顶点和索引缓冲区
    VkBuffer vertexBuffers[] = {satelliteVertexBuffer.buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, satelliteIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    
    // 渲染每个可见的卫星
    for (const auto& [satelliteId, satelliteData] : satellites) {
        if (!satelliteData.visible) {
            continue;
        }
        
        // 计算模型矩阵
        glm::mat4 model = glm::translate(glm::mat4(1.0f), satelliteData.state.position);
        model = glm::scale(model, glm::vec3(satelliteData.state.scale));
        // 应用缩放变换，使卫星可见
        
        // 推送模型矩阵
        vkCmdPushConstants(commandBuffer, satellitePipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &model);
        
        vkCmdDrawIndexed(commandBuffer, satelliteIndexCount, 1, 0, 0, 0);
    }
}

/**
 * @brief 更新统一缓冲区
 * @param frameIndex 当前帧索引
 * @param camera 相机参数
 */
void OrbitRenderer::updateUniformBuffer(uint32_t frameIndex, const CameraParams& camera) {
    OrbitUniformBufferObject ubo{};
    ubo.view = glm::lookAt(camera.position, camera.target, camera.up);
    float aspect = vulkanRenderer.getSwapChainExtent().width / (float)vulkanRenderer.getSwapChainExtent().height;
    ubo.proj = glm::perspective(glm::radians(camera.fov), aspect, camera.nearPlane, camera.farPlane);
    ubo.proj[1][1] *= -1; // Vulkan Y 轴翻转

    ubo.cameraPos = camera.position;
    // 假设 vulkanRenderer 提供了获取运行时间的方法
    ubo.currentTime = static_cast<float>(vulkanRenderer.getElapsedTime()); 
    ubo.orbitAlpha = 1.0f; // 默认透明度，后续可由参数控制
    ubo.pointSize = 2.0f; // 默认点大小，后续可由参数控制

    void* data;
    vkMapMemory(vulkanRenderer.getDevice(), uniformBuffers[frameIndex].memory, 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(vulkanRenderer.getDevice(), uniformBuffers[frameIndex].memory);
}

/**
 * @brief 清理资源
 */
void OrbitRenderer::cleanup() {
    VkDevice device = vulkanRenderer.getDevice();
    
    if (device != VK_NULL_HANDLE) {
        // 清理描述符池
        if (descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            descriptorPool = VK_NULL_HANDLE;
        }
        
        // 清理统一缓冲区
        for (auto& buffer : uniformBuffers) {
            if (buffer.buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, buffer.buffer, nullptr);
                vkFreeMemory(device, buffer.memory, nullptr);
            }
        }
        uniformBuffers.clear();
        
        // 清理管线
        if (satellitePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, satellitePipeline, nullptr);
            satellitePipeline = VK_NULL_HANDLE;
        }
        
        if (orbitPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, orbitPipeline, nullptr);
            orbitPipeline = VK_NULL_HANDLE;
        }
        
        if (satellitePipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, satellitePipelineLayout, nullptr);
            satellitePipelineLayout = VK_NULL_HANDLE;
        }
        
        if (orbitPipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, orbitPipelineLayout, nullptr);
            orbitPipelineLayout = VK_NULL_HANDLE;
        }
        
        // 清理描述符集布局
        if (descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            descriptorSetLayout = VK_NULL_HANDLE;
        }
        
        // 清理缓冲区
        if (satelliteIndexBuffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, satelliteIndexBuffer.buffer, nullptr);
            vkFreeMemory(device, satelliteIndexBuffer.memory, nullptr);
            satelliteIndexBuffer = {};
        }
        
        if (satelliteVertexBuffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, satelliteVertexBuffer.buffer, nullptr);
            vkFreeMemory(device, satelliteVertexBuffer.memory, nullptr);
            satelliteVertexBuffer = {};
        }
        
        if (orbitVertexBuffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, orbitVertexBuffer.buffer, nullptr);
            vkFreeMemory(device, orbitVertexBuffer.memory, nullptr);
            orbitVertexBuffer = {};
        }
    }
    
    // 清理数据
    orbits.clear();
    satellites.clear();
}

// namespace j2_orbit_visualization

/**
 * @brief 创建统一缓冲区
 * @return VisualizationError 创建结果
 * @exception VisualizationError::OutOfMemory 若内存分配失败
 */
VisualizationError OrbitRenderer::createUniformBuffers() {
    // 为每帧创建一个 UBO
    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkDeviceSize bufferSize = sizeof(OrbitUniformBufferObject);

    for (size_t i = 0; i < uniformBuffers.size(); ++i) {
        auto result = vulkanRenderer.createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            uniformBuffers[i]
        );
        if (result != VisualizationError::Success) {
            return result;
        }
    }

    return VisualizationError::Success;
}

/**
 * @brief 创建描述符池
 * @return VisualizationError 创建结果
 * @exception VisualizationError::VulkanError Vulkan API 调用失败
 */
VisualizationError OrbitRenderer::createDescriptorPool() {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = MAX_FRAMES_IN_FLIGHT;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;

    if (vkCreateDescriptorPool(vulkanRenderer.getDevice(), &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        return VisualizationError::VulkanError;
    }

    return VisualizationError::Success;
}

/**
 * @brief 创建描述符集
 * @return VisualizationError 创建结果
 * @exception VisualizationError::VulkanError Vulkan API 调用失败
 */
VisualizationError OrbitRenderer::createDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(vulkanRenderer.getDevice(), &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        return VisualizationError::VulkanError;
    }

    for (size_t i = 0; i < descriptorSets.size(); ++i) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i].buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(OrbitUniformBufferObject);

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = descriptorSets[i];
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(vulkanRenderer.getDevice(), 1, &descriptorWrite, 0, nullptr);
    }

    return VisualizationError::Success;
}

/**
 * @brief 清除所有轨道（不影响卫星）
 */
void OrbitRenderer::clearOrbits() {
    orbits.clear();
    updateOrbitVertexBuffer();
}

/**
 * @brief 清除所有卫星（不影响轨道）
 */
void OrbitRenderer::clearSatellites() {
    satellites.clear();
}

} // namespace j2_orbit_visualization