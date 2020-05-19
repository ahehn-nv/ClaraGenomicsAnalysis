/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "batched_device_matrices.cuh"
#include "hirschberg_myers_gpu.cuh"

namespace claragenomics
{

namespace cudaaligner
{

namespace testing
{

using WordType = hirschbergmyers::WordType;

void launch_myers_preprocess_kernel(int32_t griddim, int32_t blockdim, batched_device_matrices<WordType>::device_interface* batched_query_pattern, char const* query, int32_t query_size);

void launch_myers_get_query_pattern_test_kernel(int32_t griddim, int32_t blockdim, int32_t n_words, WordType* result, batched_device_matrices<WordType>::device_interface* batched_query_pattern, int32_t idx, char x, bool reverse);

} // namespace testing
} // namespace cudaaligner
} // namespace claragenomics
