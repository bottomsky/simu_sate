#include "earth_renderer.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <array>
#include <string>

// STB Image 库实现
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace j2_orbit_visualization {

/**
 * @brief 构造函数
 * @param renderer Vulkan 渲染器引用
 */
EarthRenderer::EarthRenderer(VulkanRenderer& renderer)
    : vulkanRenderer(renderer), indexCount(0), textureSampler(VK_NULL_HANDLE),
      descriptorSetLayout(VK_NULL_HANDLE), pipelineLayout(VK_NULL_HANDLE),
      graphicsPipeline(VK_NULL_HANDLE), descriptorPool(VK_NULL_HANDLE) {
    
    // 初始化缓冲区和图像结构
    vertexBuffer = {};
    indexBuffer = {};
    earthTexture = {};
    normalMap = {};
}

/**
 * @brief 析构函数
 * 清理地球渲染相关资源
 */
EarthRenderer::~EarthRenderer() {
    cleanup();
}

/**
 * @brief 初始化地球渲染器
 * @param textureFile 地球纹理文件路径
 * @param cloudTextureFile 云层纹理文件路径（可选）
 * @param normalMapFile 法线贴图文件路径（可选）
 * @return VisualizationError 初始化结果
 */
VisualizationError EarthRenderer::initialize(const std::string& textureFile, 
                                           const std::string& cloudTextureFile,
                                           const std::string& normalMapFile) {
    // 生成球体几何数据
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    generateSphereGeometry(vertices, indices);
    
    // 调试输出：显示生成的几何数据信息
    std::cout << "Earth mesh generated:" << std::endl;
    std::cout << "  Vertices: " << vertices.size() << std::endl;
    std::cout << "  Indices: " << indices.size() << std::endl;
    std::cout << "  Triangles: " << indices.size() / 3 << std::endl;
    
    // 创建顶点和索引缓冲区
    auto result = createVertexBuffer(vertices);
    if (result != VisualizationError::Success) {
        return result;
    }
    
    result = createIndexBuffer(indices);
    if (result != VisualizationError::Success) {
        return result;
    }
    
    // 创建地球表面纹理
    if (!textureFile.empty()) {
        result = createTextureImage(textureFile, earthTexture);
        if (result != VisualizationError::Success) {
            std::cout << "Warning: Failed to load earth texture, using default color" << std::endl;
        }
    }
    
    // 创建云层纹理
    if (!cloudTextureFile.empty()) {
        result = createTextureImage(cloudTextureFile, cloudTexture);
        if (result != VisualizationError::Success) {
            std::cout << "Warning: Failed to load cloud texture, using default transparent clouds" << std::endl;
        }
    }
    
    // 创建法线贴图（如果提供了法线贴图文件）
    if (!normalMapFile.empty()) {
        result = createTextureImage(normalMapFile, normalMap);
        if (result != VisualizationError::Success) {
            std::cout << "Warning: Failed to load normal map, using default normals" << std::endl;
        }
    }
    
    // 创建纹理采样器
    result = createTextureSampler();
    if (result != VisualizationError::Success) {
        return result;
    }
    
    // 创建描述符集布局
    result = createDescriptorSetLayout();
    if (result != VisualizationError::Success) {
        return result;
    }
    
    // 创建图形管线
    result = createGraphicsPipeline();
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
 * @brief 渲染地球
 * @param commandBuffer 命令缓冲区
 * @param renderParams 地球渲染参数
 * @param camera 相机参数
 * @return VisualizationError 渲染结果
 */
VisualizationError EarthRenderer::render(VkCommandBuffer commandBuffer, 
                                       const EarthRenderParams& renderParams,
                                       const CameraParams& camera) {
    // 更新统一缓冲区
    updateUniformBuffer(vulkanRenderer.getCurrentFrameIndex(), renderParams, camera);
    
    // 绑定图形管线
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
    
    // 绑定顶点缓冲区
    VkBuffer vertexBuffers[] = {vertexBuffer.buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    
    // 绑定索引缓冲区
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    
    // 绑定描述符集
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipelineLayout, 0, 1, &descriptorSets[vulkanRenderer.getCurrentFrameIndex()], 0, nullptr);
    
    // 绘制地球
    vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
    
    return VisualizationError::Success;
}

/**
 * @brief 更新地球渲染参数
 * @param params 新的渲染参数
 */
void EarthRenderer::updateRenderParams(const EarthRenderParams& params) {
    // 这里可以根据需要更新渲染参数
    // 例如更新光照、材质属性等
}

/**
 * @brief 生成球体几何数据
 * @param vertices 输出顶点数据
 * @param indices 输出索引数据
 */
void EarthRenderer::generateSphereGeometry(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
    vertices.clear();
    indices.clear();
    
    const float radius = 1.0f; // 单位球体，实际大小通过变换矩阵控制
    
    // 生成顶点
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
            
            // 纹理坐标（翻转Y轴以正确映射地球贴图）
            vertex.texCoord.x = (float)lon / longitudeSegments;
            vertex.texCoord.y = 1.0f - (float)lat / latitudeSegments;
            
            vertices.push_back(vertex);
        }
    }
    
    // 生成索引（逆时针顺序以匹配 VK_FRONT_FACE_COUNTER_CLOCKWISE）
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
    
    indexCount = static_cast<uint32_t>(indices.size());
}

/**
 * @brief 创建顶点缓冲区
 * @param vertices 顶点数据
 * @return VisualizationError 创建结果
 */
VisualizationError EarthRenderer::createVertexBuffer(const std::vector<Vertex>& vertices) {
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    
    // 创建暂存缓冲区
    VulkanBuffer stagingBuffer;
    auto result = vulkanRenderer.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                            stagingBuffer);
    if (result != VisualizationError::Success) {
        return result;
    }
    
    // 复制顶点数据到暂存缓冲区
    void* data;
    vkMapMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(vulkanRenderer.getDevice(), stagingBuffer.memory);
    
    // 创建顶点缓冲区
    result = vulkanRenderer.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer);
    if (result != VisualizationError::Success) {
        vkDestroyBuffer(vulkanRenderer.getDevice(), stagingBuffer.buffer, nullptr);
        vkFreeMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, nullptr);
        return result;
    }
    
    // 复制数据从暂存缓冲区到顶点缓冲区
    vulkanRenderer.copyBuffer(stagingBuffer.buffer, vertexBuffer.buffer, bufferSize);
    
    // 清理暂存缓冲区
    vkDestroyBuffer(vulkanRenderer.getDevice(), stagingBuffer.buffer, nullptr);
    vkFreeMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, nullptr);
    
    return VisualizationError::Success;
}

/**
 * @brief 创建索引缓冲区
 * @param indices 索引数据
 * @return VisualizationError 创建结果
 */
VisualizationError EarthRenderer::createIndexBuffer(const std::vector<uint32_t>& indices) {
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
    
    // 创建暂存缓冲区
    VulkanBuffer stagingBuffer;
    auto result = vulkanRenderer.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                            stagingBuffer);
    if (result != VisualizationError::Success) {
        return result;
    }
    
    // 复制索引数据到暂存缓冲区
    void* data;
    vkMapMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t)bufferSize);
    vkUnmapMemory(vulkanRenderer.getDevice(), stagingBuffer.memory);
    
    // 创建索引缓冲区
    result = vulkanRenderer.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer);
    if (result != VisualizationError::Success) {
        vkDestroyBuffer(vulkanRenderer.getDevice(), stagingBuffer.buffer, nullptr);
        vkFreeMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, nullptr);
        return result;
    }
    
    // 复制数据从暂存缓冲区到索引缓冲区
    vulkanRenderer.copyBuffer(stagingBuffer.buffer, indexBuffer.buffer, bufferSize);
    
    // 清理暂存缓冲区
    vkDestroyBuffer(vulkanRenderer.getDevice(), stagingBuffer.buffer, nullptr);
    vkFreeMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, nullptr);
    
    return VisualizationError::Success;
}

/**
 * @brief 创建纹理采样器
 * @return VisualizationError 创建结果
 */
VisualizationError EarthRenderer::createTextureSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    
    if (vkCreateSampler(vulkanRenderer.getDevice(), &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
        return VisualizationError::VulkanError;
    }
    
    return VisualizationError::Success;
}

/**
 * @brief 创建描述符集布局
 * @return VisualizationError 创建结果
 */
VisualizationError EarthRenderer::createDescriptorSetLayout() {
    // 统一缓冲区绑定
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    
    // 地球表面纹理采样器绑定
    VkDescriptorSetLayoutBinding earthSamplerLayoutBinding{};
    earthSamplerLayoutBinding.binding = 1;
    earthSamplerLayoutBinding.descriptorCount = 1;
    earthSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    earthSamplerLayoutBinding.pImmutableSamplers = nullptr;
    earthSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    
    // 云层纹理采样器绑定
    VkDescriptorSetLayoutBinding cloudSamplerLayoutBinding{};
    cloudSamplerLayoutBinding.binding = 2;
    cloudSamplerLayoutBinding.descriptorCount = 1;
    cloudSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    cloudSamplerLayoutBinding.pImmutableSamplers = nullptr;
    cloudSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    
    std::array<VkDescriptorSetLayoutBinding, 3> bindings = {uboLayoutBinding, earthSamplerLayoutBinding, cloudSamplerLayoutBinding};
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    
    if (vkCreateDescriptorSetLayout(vulkanRenderer.getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        return VisualizationError::VulkanError;
    }
    
    return VisualizationError::Success;
}

/**
 * @brief 更新统一缓冲区
 * @param frameIndex 当前帧索引
 * @param renderParams 渲染参数
 * @param camera 相机参数
 */
void EarthRenderer::updateUniformBuffer(uint32_t frameIndex, const EarthRenderParams& renderParams, 
                                      const CameraParams& camera) {
    EarthUniformBufferObject ubo{};
    
    // 模型矩阵：地球的变换（包括自转和缩放）
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::scale(model, glm::vec3(renderParams.radius)); // 使用参数中的半径
    model = glm::rotate(model, glm::radians(rotationAngle), glm::vec3(0.0f, 1.0f, 0.0f));
    
    // 视图矩阵
    glm::mat4 view = glm::lookAt(camera.position, camera.target, camera.up);
    
    // 投影矩阵
    float aspect = vulkanRenderer.getSwapChainExtent().width / (float)vulkanRenderer.getSwapChainExtent().height;
    glm::mat4 proj = glm::perspective(glm::radians(camera.fov), aspect, camera.nearPlane, camera.farPlane);
    proj[1][1] *= -1; // GLM 为 OpenGL 设计，需要翻转 Y 轴
    
    ubo.model = model;
    ubo.view = view;
    ubo.proj = proj;
    ubo.normalMatrix = glm::transpose(glm::inverse(model));
    ubo.lightPos = glm::vec3(1.0f, 1.0f, 1.0f); // 默认光源方向
    ubo.viewPos = camera.position;
    
    // 复制数据到统一缓冲区
    void* data;
    vkMapMemory(vulkanRenderer.getDevice(), uniformBuffers[frameIndex].memory, 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(vulkanRenderer.getDevice(), uniformBuffers[frameIndex].memory);
}

/**
 * @brief 清理资源
 */
void EarthRenderer::cleanup() {
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
        
        // 清理图形管线
        if (graphicsPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, graphicsPipeline, nullptr);
            graphicsPipeline = VK_NULL_HANDLE;
        }
        
        if (pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            pipelineLayout = VK_NULL_HANDLE;
        }
        
        // 清理描述符集布局
        if (descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            descriptorSetLayout = VK_NULL_HANDLE;
        }
        
        // 清理纹理采样器
        if (textureSampler != VK_NULL_HANDLE) {
            vkDestroySampler(device, textureSampler, nullptr);
            textureSampler = VK_NULL_HANDLE;
        }
        
        // 清理纹理
        if (earthTexture.image != VK_NULL_HANDLE) {
            vkDestroyImageView(device, earthTexture.view, nullptr);
            vkDestroyImage(device, earthTexture.image, nullptr);
            vkFreeMemory(device, earthTexture.memory, nullptr);
            earthTexture = {};
        }
        
        if (cloudTexture.image != VK_NULL_HANDLE) {
            vkDestroyImageView(device, cloudTexture.view, nullptr);
            vkDestroyImage(device, cloudTexture.image, nullptr);
            vkFreeMemory(device, cloudTexture.memory, nullptr);
            cloudTexture = {};
        }
        
        if (normalMap.image != VK_NULL_HANDLE) {
            vkDestroyImageView(device, normalMap.view, nullptr);
            vkDestroyImage(device, normalMap.image, nullptr);
            vkFreeMemory(device, normalMap.memory, nullptr);
            normalMap = {};
        }
        
        // 清理缓冲区
        if (indexBuffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, indexBuffer.buffer, nullptr);
            vkFreeMemory(device, indexBuffer.memory, nullptr);
            indexBuffer = {};
        }
        
        if (vertexBuffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, vertexBuffer.buffer, nullptr);
            vkFreeMemory(device, vertexBuffer.memory, nullptr);
            vertexBuffer = {};
        }
    }
}

/**
 * @brief 创建图形管线
 * @return VisualizationError 创建结果
 */
VisualizationError EarthRenderer::createGraphicsPipeline() {
    auto vertShaderCode = vulkanRenderer.readFile(std::string(SHADER_DIR) + "/earth.vert.spv");
    auto fragShaderCode = vulkanRenderer.readFile(std::string(SHADER_DIR) + "/earth.frag.spv");

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

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)vulkanRenderer.getSwapChainExtent().width;
    viewport.height = (float)vulkanRenderer.getSwapChainExtent().height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = vulkanRenderer.getSwapChainExtent();

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

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
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(vulkanRenderer.getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
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
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = vulkanRenderer.getRenderPass();
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(vulkanRenderer.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(vulkanRenderer.getDevice(), pipelineLayout, nullptr);
        vkDestroyShaderModule(vulkanRenderer.getDevice(), fragShaderModule, nullptr);
        vkDestroyShaderModule(vulkanRenderer.getDevice(), vertShaderModule, nullptr);
        return VisualizationError::VulkanError;
    }

    vkDestroyShaderModule(vulkanRenderer.getDevice(), fragShaderModule, nullptr);
    vkDestroyShaderModule(vulkanRenderer.getDevice(), vertShaderModule, nullptr);

    return VisualizationError::Success;
}

/**
 * @brief 创建描述符池
 * @return VisualizationError 创建结果
 * @exception VisualizationError::VulkanError Vulkan API 调用失败
 */
VisualizationError EarthRenderer::createDescriptorPool() {
    std::vector<VkDescriptorPoolSize> poolSizes{};
    
    // 统一缓冲区描述符
    VkDescriptorPoolSize uboPoolSize{};
    uboPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboPoolSize.descriptorCount = static_cast<uint32_t>(vulkanRenderer.getMaxFramesInFlight());
    poolSizes.push_back(uboPoolSize);
    
    // 纹理采样器描述符（地球表面和云层纹理）
    VkDescriptorPoolSize samplerPoolSize{};
    samplerPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerPoolSize.descriptorCount = static_cast<uint32_t>(vulkanRenderer.getMaxFramesInFlight() * 2); // 地球表面和云层纹理
    poolSizes.push_back(samplerPoolSize);

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(vulkanRenderer.getMaxFramesInFlight());

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
VisualizationError EarthRenderer::createDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(vulkanRenderer.getMaxFramesInFlight(), descriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(vulkanRenderer.getMaxFramesInFlight());
    if (vkAllocateDescriptorSets(vulkanRenderer.getDevice(), &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        return VisualizationError::VulkanError;
    }

    for (size_t i = 0; i < descriptorSets.size(); ++i) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i].buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(EarthUniformBufferObject);

        VkDescriptorImageInfo earthImageInfo{};
        earthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        earthImageInfo.imageView = earthTexture.view;
        earthImageInfo.sampler = textureSampler;
        
        VkDescriptorImageInfo cloudImageInfo{};
        cloudImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        cloudImageInfo.imageView = cloudTexture.view;
        cloudImageInfo.sampler = textureSampler;

        std::array<VkWriteDescriptorSet, 3> descriptorWrites{};

        // 统一缓冲区
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        // 地球表面纹理
        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &earthImageInfo;
        
        // 云层纹理
        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = descriptorSets[i];
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pImageInfo = &cloudImageInfo;

        vkUpdateDescriptorSets(vulkanRenderer.getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }

    return VisualizationError::Success;
}

/**
 * @brief 创建纹理图像
 * @param textureFile 纹理文件路径
 * @param image 输出图像结构
 * @return VisualizationError 创建结果
 */
VisualizationError EarthRenderer::createTextureImage(const std::string& textureFile, VulkanImage& image) {
    // 使用 stb_image 加载图像
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(textureFile.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    
    if (!pixels) {
        std::cerr << "Failed to load texture image: " << textureFile << std::endl;
        std::cerr << "STB Error: " << stbi_failure_reason() << std::endl;
        return VisualizationError::TEXTURE_LOADING_FAILED;
    }
    
    std::cout << "Loaded texture: " << textureFile << " (" << texWidth << "x" << texHeight << ", " << texChannels << " channels)" << std::endl;
    
    const VkDeviceSize imageSize = texWidth * texHeight * 4; // 强制使用 RGBA
    
    // 创建暂存缓冲区
    VulkanBuffer stagingBuffer;
    auto result = vulkanRenderer.createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                            stagingBuffer);
    if (result != VisualizationError::Success) {
        stbi_image_free(pixels);
        return result;
    }
    
    // 复制像素数据到暂存缓冲区
    void* data;
    vkMapMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(vulkanRenderer.getDevice(), stagingBuffer.memory);
    
    // 释放 stb_image 分配的内存
    stbi_image_free(pixels);
    
    // 创建图像
    result = vulkanRenderer.createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB,
                                       VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image);
    if (result != VisualizationError::Success) {
        vkDestroyBuffer(vulkanRenderer.getDevice(), stagingBuffer.buffer, nullptr);
        vkFreeMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, nullptr);
        return result;
    }
    
    // 转换图像布局并复制数据
    vulkanRenderer.transitionImageLayout(image.image, VK_FORMAT_R8G8B8A8_SRGB,
                                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vulkanRenderer.copyBufferToImage(stagingBuffer.buffer, image.image, texWidth, texHeight);
    vulkanRenderer.transitionImageLayout(image.image, VK_FORMAT_R8G8B8A8_SRGB,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    
    // 清理暂存缓冲区
    vkDestroyBuffer(vulkanRenderer.getDevice(), stagingBuffer.buffer, nullptr);
    vkFreeMemory(vulkanRenderer.getDevice(), stagingBuffer.memory, nullptr);
    
    // 创建图像视图
    image.view = vulkanRenderer.createImageView(image.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
    if (image.view == VK_NULL_HANDLE) {
        return VisualizationError::VulkanError;
    }
    
    return VisualizationError::Success;
}

/**
 * @brief 创建统一缓冲区
 * @return VisualizationError 创建结果
 */
VisualizationError EarthRenderer::createUniformBuffers() {
    VkDeviceSize bufferSize = sizeof(EarthUniformBufferObject);
    
    uniformBuffers.resize(vulkanRenderer.getMaxFramesInFlight());
    
    for (size_t i = 0; i < uniformBuffers.size(); i++) {
        auto result = vulkanRenderer.createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                uniformBuffers[i]);
        if (result != VisualizationError::Success) {
            // 清理已创建的缓冲区
            for (size_t j = 0; j < i; j++) {
                vkDestroyBuffer(vulkanRenderer.getDevice(), uniformBuffers[j].buffer, nullptr);
                vkFreeMemory(vulkanRenderer.getDevice(), uniformBuffers[j].memory, nullptr);
            }
            uniformBuffers.clear();
            return result;
        }
    }
    
    return VisualizationError::Success;
}

} // namespace j2_orbit_visualization