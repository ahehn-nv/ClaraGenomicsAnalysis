/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <claragenomics/cudaaligner/aligner.hpp>

#include "aligner_global_hirschberg_myers.hpp"
#include "aligner_global_myers_banded.hpp"

namespace claragenomics
{

namespace cudaaligner
{

std::unique_ptr<Aligner> create_aligner(
    int32_t max_query_length, int32_t max_target_length,
    int32_t max_alignments, AlignmentType type,
    DefaultDeviceAllocator allocator, cudaStream_t stream, int32_t device_id)
{
    if (type == AlignmentType::global_alignment)
    {
        return std::make_unique<AlignerGlobalHirschbergMyers>(max_query_length, max_target_length, max_alignments, allocator, stream, device_id);
    }
    else
    {
        throw std::runtime_error("Aligner for specified type not implemented yet.");
    }
}

std::unique_ptr<Aligner> create_aligner(
    int32_t max_query_length, int32_t max_target_length,
    int32_t max_alignments, AlignmentType type,
    cudaStream_t stream, int32_t device_id, int64_t max_device_memory_allocator_caching_size)
{
    if (max_device_memory_allocator_caching_size < -1)
    {
        throw std::invalid_argument("max_device_memory_allocator_caching_size has to be either -1 (=all available GPU memory) or greater or equal than 0.");
    }
#ifdef CGA_ENABLE_CACHING_ALLOCATOR
    // uses CachingDeviceAllocator
    if (max_device_memory_allocator_caching_size == -1)
    {
        max_device_memory_allocator_caching_size = claragenomics::cudautils::find_largest_contiguous_device_memory_section();
        if (max_device_memory_allocator_caching_size == 0)
        {
            throw std::runtime_error("No memory available for caching");
        }
    }
    claragenomics::DefaultDeviceAllocator allocator(max_device_memory_allocator_caching_size);
#else
    // uses CudaMallocAllocator
    claragenomics::DefaultDeviceAllocator allocator;
#endif
    return create_aligner(max_query_length, max_target_length, max_alignments, type, allocator, stream, device_id);
}

std::unique_ptr<Aligner> create_aligner(
    const AlignmentType type,
    const int32_t max_bandwidth,
    cudaStream_t stream,
    const int32_t device_id,
    DefaultDeviceAllocator allocator,
    const int64_t max_device_memory)
{
    if (type == AlignmentType::global_alignment)
    {
        return std::make_unique<AlignerGlobalMyersBanded>(max_device_memory, max_bandwidth, allocator, stream, device_id);
    }
    else
    {
        throw std::runtime_error("Aligner for specified type not implemented yet.");
    }
}

std::unique_ptr<Aligner> create_aligner(
    const AlignmentType type,
    const int32_t max_bandwidth,
    cudaStream_t stream,
    const int32_t device_id,
    int64_t max_device_memory)
{
    if (max_device_memory < -1)
    {
        throw std::invalid_argument("max_device_memory has to be either -1 (=all available GPU memory) or greater or equal than 0.");
    }
#ifdef CGA_ENABLE_CACHING_ALLOCATOR
    // uses CachingDeviceAllocator
    if (max_device_memory == -1)
    {
        max_device_memory = claragenomics::cudautils::find_largest_contiguous_device_memory_section();
        if (max_device_memory == 0)
        {
            throw std::runtime_error("No memory available for caching");
        }
    }
    claragenomics::DefaultDeviceAllocator allocator(max_device_memory);
#else
    // uses CudaMallocAllocator
    claragenomics::DefaultDeviceAllocator allocator;
#endif
    return create_aligner(type, max_bandwidth, stream, device_id, allocator, -1);
}

} // namespace cudaaligner
} // namespace claragenomics
