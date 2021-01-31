
/*
* Copyright 2019-2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

#include <claraparabricks/genomeworks/utils/allocator.hpp>
#include <claraparabricks/genomeworks/cudaaligner/alignment.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

void convert_to_cigar_gpu(std::vector<std::shared_ptr<Alignment>>& alignments_output, int8_t* alignment_sequences, int32_t* alignment_lengths, int64_t* alignment_starts, DefaultDeviceAllocator allocator, cudaStream_t stream);

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
