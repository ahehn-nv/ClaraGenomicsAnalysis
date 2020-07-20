/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <claragenomics/cudaaligner/cudaaligner.hpp>
#include <claragenomics/utils/allocator.hpp>

#include <memory>
#include <vector>
#include <cuda_runtime_api.h>

namespace claragenomics
{

namespace cudaaligner
{

// Forward declaration of Alignment class.
class Alignment;

/// \addtogroup cudaaligner
/// \{

/// \class Aligner
/// CUDA Alignment object
class Aligner
{
public:
    /// \brief Virtual destructor for Aligner.
    virtual ~Aligner() = default;

    /// \brief Launch CUDA accelerated alignment
    ///
    /// Perform alignment on all Alignment objects previously
    /// inserted. This is an async call, and returns before alignment
    /// is fully finished. To sync the alignments, refer to the
    /// sync_alignments() call;
    /// To
    virtual StatusType align_all() = 0;

    /// \brief Waits for CUDA accelerated alignment to finish
    ///
    /// Blocking call that waits for all the alignments scheduled
    /// on the GPU to come to completion.
    virtual StatusType sync_alignments() = 0;

    /// \brief Add new alignment object. Only strings with characters
    ///        from the alphabet [ACGT] are guaranteed to provide correct results.
    ///
    /// \param query Query string
    /// \param query_length  Query string length
    /// \param target Target string
    /// \param target_length Target string length
    /// \param reverse_complement_query Reverse complement the query string
    /// \param reverse_complement_target Reverse complement the target string
    virtual StatusType add_alignment(const char* query, int32_t query_length, const char* target, int32_t target_length,
                                     bool reverse_complement_query = false, bool reverse_complement_target = false) = 0;

    /// \brief Return the computed alignments.
    ///
    /// \return Vector of Alignments.
    virtual const std::vector<std::shared_ptr<Alignment>>& get_alignments() const = 0;

    /// \brief Reset aligner object.
    virtual void reset() = 0;
};

/// \brief Created Aligner object
///
/// \param max_query_length Maximum length of query string
/// \param max_target_length Maximum length of target string
/// \param max_alignments Maximum number of alignments to be performed
/// \param type Type of aligner to construct
/// \param allocator Allocator to use for internal device memory allocations
/// \param stream CUDA Stream used for GPU interaction of the object
/// \param device_id GPU device ID to run all CUDA operations on
///
/// \return Unique pointer to Aligner object
std::unique_ptr<Aligner> create_aligner(int32_t max_query_length, int32_t max_target_length, int32_t max_alignments, AlignmentType type, DefaultDeviceAllocator allocator, cudaStream_t stream, int32_t device_id);
// FIXME

/// \brief Created Aligner object
///
/// \param max_query_length Maximum length of query string
/// \param max_target_length Maximum length of target string
/// \param max_alignments Maximum number of alignments to be performed
/// \param type Type of aligner to construct
/// \param stream CUDA Stream used for GPU interaction of the object
/// \param device_id GPU device ID to run all CUDA operations on
/// \param max_device_memory_allocator_caching_size Maximum amount of device memory to use for cached memory allocations the cudaaligner instance. max_device_memory_allocator_caching_size = -1 (default) means all available device memory. This parameter is ignored if the SDK is compiled for non-caching allocators.
///
/// \return Unique pointer to Aligner object
std::unique_ptr<Aligner> create_aligner(int32_t max_query_length, int32_t max_target_length, int32_t max_alignments, AlignmentType type, cudaStream_t stream, int32_t device_id, int64_t max_device_memory_allocator_caching_size = -1);
// FIXME

/// \brief Created Aligner object
///
/// \param type Type of aligner to construct
/// \param max_bandwidth Maximum bandwidth for the Ukkonen band
/// \param stream CUDA Stream used for GPU interaction of the object
/// \param device_id GPU device ID to run all CUDA operations on
/// \param allocator Allocator to use for internal device memory allocations
/// \param max_device_memory Maximum amount of device memory to use from passed in allocator in bytes (-1 for all available memory)
///
/// \return Unique pointer to Aligner object
std::unique_ptr<Aligner> create_aligner(AlignmentType type, int32_t max_bandwidth, cudaStream_t stream, int32_t device_id, DefaultDeviceAllocator allocator, int64_t max_device_memory);

/// \brief Created Aligner object
///
/// \param type Type of aligner to construct
/// \param max_bandwidth Maximum bandwidth for the Ukkonen band
/// \param stream CUDA Stream used for GPU interaction of the object
/// \param device_id GPU device ID to run all CUDA operations on
/// \param max_device_memory Maximum amount of device memory used in bytes (-1 (default) for all available memory).
///
/// \return Unique pointer to Aligner object
std::unique_ptr<Aligner> create_aligner(AlignmentType type, int32_t max_bandwidth, cudaStream_t stream, int32_t device_id, int64_t max_device_memory = -1);
/// \}
} // namespace cudaaligner
} // namespace claragenomics
