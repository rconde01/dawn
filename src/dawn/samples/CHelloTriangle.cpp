// Copyright 2017 The Dawn & Tint Authors
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "dawn/samples/SampleUtils.h"

#include "dawn/utils/SystemUtils.h"
#include "dawn/utils/WGPUHelpers.h"

WGPUDevice device;
WGPUQueue queue;
WGPUSwapChain swapchain;
WGPURenderPipeline pipeline;
WGPUBuffer indirectBuffer;

WGPUTextureFormat swapChainFormat;

void init() {
    device = CreateCppDawnDevice().MoveToCHandle();
    queue = wgpuDeviceGetQueue(device);
    swapchain = GetSwapChain().MoveToCHandle();
    swapChainFormat = static_cast<WGPUTextureFormat>(GetPreferredSwapChainTextureFormat());

    const char* vs = R"(
        @vertex fn main(
            @builtin(vertex_index) VertexIndex : u32,
            @builtin(instance_index) InstanceIndex: u32
        ) -> @builtin(position) vec4f {
            let offset = f32(InstanceIndex)*0.2;

            var pos = array(
                vec2f( 0.0 + offset,  0.5),
                vec2f(-0.5 + offset, -0.5),
                vec2f( 0.5 + offset, -0.5)
            );
            return vec4f(pos[VertexIndex], 0.0, 1.0);
        })";
    WGPUShaderModule vsModule = dawn::utils::CreateShaderModule(device, vs).MoveToCHandle();

    const char* fs = R"(
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 1.0);
        })";
    WGPUShaderModule fsModule = dawn::utils::CreateShaderModule(device, fs).MoveToCHandle();

    {
        WGPURenderPipelineDescriptor descriptor = {};

        // Fragment state
        WGPUBlendState blend = {};
        blend.color.operation = WGPUBlendOperation_Add;
        blend.color.srcFactor = WGPUBlendFactor_One;
        blend.color.dstFactor = WGPUBlendFactor_One;
        blend.alpha.operation = WGPUBlendOperation_Add;
        blend.alpha.srcFactor = WGPUBlendFactor_One;
        blend.alpha.dstFactor = WGPUBlendFactor_One;

        WGPUColorTargetState colorTarget = {};
        colorTarget.format = swapChainFormat;
        colorTarget.blend = &blend;
        colorTarget.writeMask = WGPUColorWriteMask_All;

        WGPUFragmentState fragment = {};
        fragment.module = fsModule;
        fragment.entryPoint = "main";
        fragment.targetCount = 1;
        fragment.targets = &colorTarget;
        descriptor.fragment = &fragment;

        // Other state
        descriptor.layout = nullptr;
        descriptor.depthStencil = nullptr;

        descriptor.vertex.module = vsModule;
        descriptor.vertex.entryPoint = "main";
        descriptor.vertex.bufferCount = 0;
        descriptor.vertex.buffers = nullptr;

        descriptor.multisample.count = 1;
        descriptor.multisample.mask = 0xFFFFFFFF;
        descriptor.multisample.alphaToCoverageEnabled = false;

        descriptor.primitive.frontFace = WGPUFrontFace_CCW;
        descriptor.primitive.cullMode = WGPUCullMode_None;
        descriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        descriptor.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;

        pipeline = wgpuDeviceCreateRenderPipeline(device, &descriptor);
    }

    wgpuShaderModuleRelease(vsModule);
    wgpuShaderModuleRelease(fsModule);

    {
        auto const indirectParamsSize = 4 * sizeof(uint32_t);

        WGPUBufferDescriptor bufferDescriptor = {};
        bufferDescriptor.mappedAtCreation = true;
        bufferDescriptor.size = 2 * indirectParamsSize;
        bufferDescriptor.usage = WGPUBufferUsage::WGPUBufferUsage_Indirect;

        indirectBuffer = wgpuDeviceCreateBuffer(device, &bufferDescriptor);

        auto indirectParams = static_cast<uint32_t*>(
            wgpuBufferGetMappedRange(indirectBuffer, 0, bufferDescriptor.size));

        indirectParams[0 * 4 + 0] = 3;
        indirectParams[0 * 4 + 1] = 1;
        indirectParams[0 * 4 + 2] = 0;
        indirectParams[0 * 4 + 3] = 0;

        indirectParams[1 * 4 + 0] = 3;
        indirectParams[1 * 4 + 1] = 1;
        indirectParams[1 * 4 + 2] = 0;
        indirectParams[1 * 4 + 3] = 1;

        wgpuBufferUnmap(indirectBuffer);
    }
}

void frame() {
    WGPUTextureView backbufferView = wgpuSwapChainGetCurrentTextureView(swapchain);
    WGPURenderPassDescriptor renderpassInfo = {};
    WGPURenderPassColorAttachment colorAttachment = {};
    {
        colorAttachment.view = backbufferView;
        colorAttachment.resolveTarget = nullptr;
        colorAttachment.clearValue = {0.0f, 0.0f, 0.0f, 0.0f};
        colorAttachment.loadOp = WGPULoadOp_Clear;
        colorAttachment.storeOp = WGPUStoreOp_Store;
        renderpassInfo.colorAttachmentCount = 1;
        renderpassInfo.colorAttachments = &colorAttachment;
        renderpassInfo.depthStencilAttachment = nullptr;
    }
    WGPUCommandBuffer commands;
    {
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);

        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &renderpassInfo);
        wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        wgpuRenderPassEncoderDrawIndirect(pass, indirectBuffer, 0);
        wgpuRenderPassEncoderDrawIndirect(pass, indirectBuffer, 4 * sizeof(uint32_t));
        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);

        commands = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuCommandEncoderRelease(encoder);
    }

    wgpuQueueSubmit(queue, 1, &commands);
    wgpuCommandBufferRelease(commands);
    wgpuSwapChainPresent(swapchain);
    wgpuTextureViewRelease(backbufferView);

    DoFlush();
}

int main(int argc, const char* argv[]) {
    if (!InitSample(argc, argv)) {
        return 1;
    }
    init();

    while (!ShouldQuit()) {
        ProcessEvents();
        frame();
        dawn::utils::USleep(16000);
    }
}
