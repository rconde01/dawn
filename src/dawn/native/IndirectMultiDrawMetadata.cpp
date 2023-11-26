// Copyright 2021 The Dawn & Tint Authors
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

#include "dawn/native/IndirectMultiDrawMetadata.h"

#include <algorithm>
#include <utility>

#include "dawn/common/Constants.h"
#include "dawn/native/IndirectMultiDrawValidationEncoder.h"
#include "dawn/native/Limits.h"
#include "dawn/native/RenderBundle.h"

namespace dawn::native {

uint64_t ComputeMaxIndirectValidationBatchOffsetRangeSize(const CombinedLimits& limits) {
    // TODO(RWC) - Should this account for duplicating base/instance
    return limits.v1.maxStorageBufferBindingSize - limits.v1.minStorageBufferOffsetAlignment;

    // We subtract kDrawIndexedIndirectSize because the maxOffset is the start offset...so we
    // need room for the actual draw
}

IndirectMultiDrawMetadata::IndexedIndirectBufferValidationInfo::IndexedIndirectBufferValidationInfo(
    BufferBase* indirectBuffer)
    : mIndirectBuffer(indirectBuffer) {}

void IndirectMultiDrawMetadata::IndexedIndirectBufferValidationInfo::AddIndirectMultiDraw(
    uint32_t maxDrawCallsPerIndirectValidationBatch,
    uint64_t maxBatchOffsetRange,
    IndirectMultiDraw draw) {
    const uint64_t newMinByte = draw.inputBufferOffset;
    const uint64_t newMaxByte =
        draw.inputBufferOffset + draw.cmd->maxDrawCount * kDrawIndirectSize - 1;
    auto it = mBatches.begin();
    while (it != mBatches.end()) {
        IndirectValidationBatch& batch = *it;
        if (batch.draws.size() >= maxDrawCallsPerIndirectValidationBatch) {
            // TODO(RWC) Batches might overlap - but i think it's ok? The minByte of each batch will
            // be ordered, but the max of a batch might be beyond the min of the next.
            // This batch is full. If its minOffset is to the right of the new offset, we can
            // just insert a new batch here.
            if (newMinByte < batch.minByte) {
                break;
            }

            // Otherwise keep looking.
            ++it;
            continue;
        }

        if (newMinByte >= batch.minByte && newMaxByte <= batch.maxByte) {
            batch.draws.push_back(std::move(draw));
            return;
        }

        // TODO(RWC) what if a single call is bigger than maxBatchOffsetRange
        const uint64_t extended_min = std::min(batch.minByte, newMinByte);
        const uint64_t extended_max = std::max(batch.maxByte, newMaxByte);

        if (extended_max - extended_min + 1 <= maxBatchOffsetRange) {
            batch.minByte = extended_min;
            batch.maxByte = extended_max;
            batch.draws.push_back(std::move(draw));
            return;
        }

        if (newMinByte < batch.minByte) {
            // We want to insert a new batch just before this one.
            break;
        }

        ++it;
    }

    IndirectValidationBatch newBatch;
    newBatch.minByte = newMinByte;
    newBatch.maxByte = newMaxByte;
    newBatch.draws.push_back(std::move(draw));

    mBatches.insert(it, std::move(newBatch));
}

void IndirectMultiDrawMetadata::IndexedIndirectBufferValidationInfo::AddBatch(
    uint32_t maxDrawCallsPerIndirectValidationBatch,
    uint64_t maxBatchOffsetRange,
    const IndirectValidationBatch& newBatch) {
    auto it = mBatches.begin();
    while (it != mBatches.end()) {
        IndirectValidationBatch& batch = *it;
        uint64_t min = std::min(newBatch.minByte, batch.minByte);
        uint64_t max = std::max(newBatch.maxByte, batch.maxByte);
        if (max - min <= maxBatchOffsetRange &&
            batch.draws.size() + newBatch.draws.size() <= maxDrawCallsPerIndirectValidationBatch) {
            // This batch fits within the limits of an existing batch. Merge it.
            batch.minByte = min;
            batch.maxByte = max;
            batch.draws.insert(batch.draws.end(), newBatch.draws.begin(), newBatch.draws.end());
            return;
        }

        if (newBatch.minByte < batch.minByte) {
            break;
        }

        ++it;
    }
    mBatches.push_back(newBatch);
}

const std::vector<IndirectMultiDrawMetadata::IndirectValidationBatch>&
IndirectMultiDrawMetadata::IndexedIndirectBufferValidationInfo::GetBatches() const {
    return mBatches;
}

IndirectMultiDrawMetadata::IndirectMultiDrawMetadata(const CombinedLimits& limits)
    : mMaxBatchOffsetRange(ComputeMaxIndirectValidationBatchOffsetRangeSize(limits)),
      mMaxDrawCallsPerBatch(ComputeMaxMultiDrawCallsPerIndirectValidationBatch(limits)) {}

IndirectMultiDrawMetadata::~IndirectMultiDrawMetadata() = default;

IndirectMultiDrawMetadata::IndirectMultiDrawMetadata(IndirectMultiDrawMetadata&&) = default;

IndirectMultiDrawMetadata& IndirectMultiDrawMetadata::operator=(IndirectMultiDrawMetadata&&) =
    default;

IndirectMultiDrawMetadata::IndexedIndirectBufferValidationInfoMap*
IndirectMultiDrawMetadata::GetIndexedIndirectBufferValidationInfo() {
    return &mIndexedIndirectBufferValidationInfo;
}

void IndirectMultiDrawMetadata::AddBundle(RenderBundleBase* bundle) {
    auto [_, inserted] = mAddedBundles.insert(bundle);
    if (!inserted) {
        return;
    }

    for (const auto& [config, validationInfo] :
         bundle->GetIndirectMultiDrawMetadata().mIndexedIndirectBufferValidationInfo) {
        auto it = mIndexedIndirectBufferValidationInfo.lower_bound(config);
        if (it != mIndexedIndirectBufferValidationInfo.end() && it->first == config) {
            // We already have batches for the same config. Merge the new ones in.
            for (const IndirectValidationBatch& batch : validationInfo.GetBatches()) {
                it->second.AddBatch(mMaxDrawCallsPerBatch, mMaxBatchOffsetRange, batch);
            }
        } else {
            mIndexedIndirectBufferValidationInfo.emplace_hint(it, config, validationInfo);
        }
    }
}

void IndirectMultiDrawMetadata::AddIndirectMultiDraw(BufferBase* indirectBuffer,
                                                     uint64_t indirectOffset,
                                                     bool duplicateBaseVertexInstance,
                                                     MultiDrawIndirectCmd* cmd) {
    const IndexedIndirectConfig config = {indirectBuffer, 0, duplicateBaseVertexInstance,
                                          DrawType::NonIndexed};
    auto it = mIndexedIndirectBufferValidationInfo.find(config);
    if (it == mIndexedIndirectBufferValidationInfo.end()) {
        auto result = mIndexedIndirectBufferValidationInfo.emplace(
            config, IndexedIndirectBufferValidationInfo(indirectBuffer));
        it = result.first;
    }

    IndirectMultiDraw draw{};
    draw.inputBufferOffset = indirectOffset;
    draw.cmd = cmd;
    it->second.AddIndirectMultiDraw(mMaxDrawCallsPerBatch, mMaxBatchOffsetRange, draw);
}

bool IndirectMultiDrawMetadata::IndexedIndirectConfig::operator<(
    const IndexedIndirectConfig& other) const {
    return std::tie(inputIndirectBuffer, numIndexBufferElements, duplicateBaseVertexInstance,
                    drawType) < std::tie(other.inputIndirectBuffer, other.numIndexBufferElements,
                                         other.duplicateBaseVertexInstance, other.drawType);
}

bool IndirectMultiDrawMetadata::IndexedIndirectConfig::operator==(
    const IndexedIndirectConfig& other) const {
    return std::tie(inputIndirectBuffer, numIndexBufferElements, duplicateBaseVertexInstance,
                    drawType) == std::tie(other.inputIndirectBuffer, other.numIndexBufferElements,
                                          other.duplicateBaseVertexInstance, other.drawType);
}

}  // namespace dawn::native
