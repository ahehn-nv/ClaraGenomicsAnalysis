/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include "aligner_global.hpp"

namespace claragenomics
{

namespace cudaaligner
{

class AlignerGlobalMyersBanded : public Aligner
{
public:
    AlignerGlobalMyersBanded(int64_t max_device_memory, int32_t max_bandwidth, DefaultDeviceAllocator allocator, cudaStream_t stream, int32_t device_id);
    ~AlignerGlobalMyersBanded() override;

    StatusType align_all() override;
    StatusType sync_alignments() override;
    void reset() override;

    StatusType add_alignment(const char* query, int32_t query_length, const char* target, int32_t target_length, bool reverse_complement_query, bool reverse_complement_target) override;

    const std::vector<std::shared_ptr<Alignment>>& get_alignments() const override
    {
        return alignments_;
    }

private:
    void reset_data();

    struct InternalData;
    std::unique_ptr<InternalData> data_;
    cudaStream_t stream_;
    int32_t device_id_;
    int32_t max_bandwidth_;
    std::vector<std::shared_ptr<Alignment>> alignments_;
};

} // namespace cudaaligner
} // namespace claragenomics
