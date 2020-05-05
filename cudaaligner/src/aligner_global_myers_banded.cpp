/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "aligner_global_myers_banded.hpp"
#include "myers_gpu.cuh"
#include "batched_device_matrices.cuh"

#include <claragenomics/utils/mathutils.hpp>

namespace claragenomics
{

namespace cudaaligner
{

namespace
{

constexpr float good_max_distance_estimate_factor = 0.1; // query has to be >=90% of target length
constexpr int32_t word_size                       = sizeof(myers::WordType) * CHAR_BIT;

int64_t compute_good_matrix_size_per_alignment(int32_t max_target_length)
{
    const int32_t query_size            = max_target_length;
    const int32_t max_distance_estimate = static_cast<int32_t>(good_max_distance_estimate_factor * max_target_length);
    const int32_t p                     = max_distance_estimate / 2;
    const int32_t band_width            = std::min(1 + 2 * p, query_size);
    const int64_t n_words_band          = ceiling_divide(band_width, word_size);
    return n_words_band * (max_target_length + 1);
}

} // namespace

struct AlignerGlobalMyersBanded::Workspace
{
    Workspace(int32_t max_alignments, int32_t max_n_words, int64_t initial_matrix_size_per_alignment, DefaultDeviceAllocator allocator, cudaStream_t stream)
        : alignment_indices_0(max_alignments, allocator, stream)
        , alignment_indices_1(max_alignments, allocator, stream)
        , n_alignments_atomics(2, allocator, stream)
        , pvs(max_alignments, initial_matrix_size_per_alignment, allocator, stream)
        , mvs(max_alignments, initial_matrix_size_per_alignment, allocator, stream)
        , scores(max_alignments, initial_matrix_size_per_alignment, allocator, stream)
        , query_patterns(max_alignments, max_n_words * 4, allocator, stream)
    {
    }
    device_buffer<int32_t> alignment_indices_0;
    device_buffer<int32_t> alignment_indices_1;
    device_buffer<int32_t> n_alignments_atomics;
    batched_device_matrices<myers::WordType> pvs;
    batched_device_matrices<myers::WordType> mvs;
    batched_device_matrices<int32_t> scores;
    batched_device_matrices<myers::WordType> query_patterns;
};

AlignerGlobalMyersBanded::AlignerGlobalMyersBanded(int32_t max_query_length, int32_t max_target_length, int32_t max_alignments, DefaultDeviceAllocator allocator, cudaStream_t stream, int32_t device_id)
    : AlignerGlobal(max_query_length, max_target_length, max_alignments, allocator, stream, device_id)
    , workspace_()
{
    scoped_device_switch dev(device_id);
    const int32_t max_words_query                   = ceiling_divide<int32_t>(max_query_length, word_size);
    const int64_t initial_matrix_size_per_alignment = compute_good_matrix_size_per_alignment(max_target_length);
    workspace_                                      = std::make_unique<Workspace>(max_alignments, max_words_query, initial_matrix_size_per_alignment, allocator, stream);
}

AlignerGlobalMyersBanded::~AlignerGlobalMyersBanded()
{
    // Keep empty destructor to keep Workspace type incomplete in the .hpp file.
}

void AlignerGlobalMyersBanded::run_alignment(int8_t* results_d, int32_t* result_lengths_d, int32_t max_result_length,
                                             const char* sequences_d, int32_t* sequence_lengths_d, int32_t* sequence_lengths_h, int32_t max_sequence_length,
                                             int32_t num_alignments, cudaStream_t stream)
{
    static_cast<void>(sequence_lengths_h);
    myers_banded_gpu(results_d, result_lengths_d, max_result_length,
                     sequences_d, sequence_lengths_d, max_sequence_length,
                     num_alignments, workspace_->n_alignments_atomics,
                     workspace_->alignment_indices_0, workspace_->alignment_indices_1,
                     workspace_->pvs, workspace_->mvs, workspace_->scores, workspace_->query_patterns,
                     stream);
}

} // namespace cudaaligner
} // namespace claragenomics
