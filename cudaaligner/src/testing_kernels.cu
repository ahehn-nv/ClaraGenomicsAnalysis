/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "testing_kernels.cuh"
#include "hirschberg_myers_gpu.cuh"
#include <claragenomics/utils/mathutils.hpp>

namespace claragenomics
{

namespace cudaaligner
{

namespace testing
{

namespace kernels
{
__global__ void myers_preprocess_kernel(batched_device_matrices<WordType>::device_interface* batched_query_pattern, char const* query, int32_t query_size)
{
    CGA_CONSTEXPR int32_t word_size            = sizeof(WordType) * CHAR_BIT;
    const int32_t n_words                      = ceiling_divide<int32_t>(query_size, word_size);
    device_matrix_view<WordType> query_pattern = batched_query_pattern->get_matrix_view(0, n_words, 8);
    hirschbergmyers::myers_preprocess(query_pattern, query, query_size);
}

__global__ void myers_get_query_pattern_test_kernel(int32_t n_words, WordType* result, batched_device_matrices<WordType>::device_interface* batched_query_pattern, int32_t idx, char x, bool reverse)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 32)
    {
        device_matrix_view<WordType> patterns = batched_query_pattern->get_matrix_view(0, n_words, 8);
        result[i]                             = hirschbergmyers::get_query_pattern(patterns, idx, i, x, reverse);
    }
}
} // namespace kernels

void launch_myers_preprocess_kernel(int32_t griddim, int32_t blockdim, batched_device_matrices<WordType>::device_interface* batched_query_pattern, char const* query, int32_t query_size)
{
    kernels::myers_preprocess_kernel<<<griddim, blockdim>>>(batched_query_pattern, query, query_size);
}

void launch_myers_get_query_pattern_test_kernel(int32_t griddim, int32_t blockdim, int32_t n_words, WordType* result, batched_device_matrices<WordType>::device_interface* batched_query_pattern, int32_t idx, char x, bool reverse)
{
    kernels::myers_get_query_pattern_test_kernel<<<griddim, blockdim>>>(n_words, result, batched_query_pattern, idx, x, reverse);
}

} // namespace testing
} // namespace cudaaligner
} // namespace claragenomics
