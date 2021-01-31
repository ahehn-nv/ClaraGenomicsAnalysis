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

#include "convert_to_cigar.cuh"
#include "alignment_impl.hpp"

#include <claraparabricks/genomeworks/utils/pinned_host_vector.hpp>
#include <claraparabricks/genomeworks/utils/device_buffer.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>
#include <thrust/reverse.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <cub/device/device_run_length_encode.cuh>
#include <algorithm>
#include <numeric>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

// This function reverses the alignment sequence
//__global__ void convert_to_cigar_kernel(int8_t* alignment_sequences, int64_t* alignment_starts, int32_t* alignment_lengths, char* cigar_strings, int64_t* cigar_starts, int32_t* cigar_lengths)
//{
//}
struct translate_to_cigar_chars : public thrust::unary_function<float, float>
{
    __host__ __device__ int8_t operator()(int8_t x) const
    {
        // CIGAR string format from http://bioinformatics.cvr.ac.uk/blog/tag/cigar-string/
        // Implementing the set of CIGAR states with =, X, D and I characters,
        // which distingishes matches and mismatches.
        switch (x)
        {
        case static_cast<int8_t>(AlignmentState::match): return '=';
        case static_cast<int8_t>(AlignmentState::mismatch): return 'X';
        case static_cast<int8_t>(AlignmentState::insertion): return 'I';
        case static_cast<int8_t>(AlignmentState::deletion): return 'D';
        default: return '!';
        }
    }
};

void convert_to_cigar_gpu(
    std::vector<std::shared_ptr<Alignment>>& alignments_out,
    int8_t* alignment_sequences,
    int32_t* alignment_lengths_host,
    int64_t* alignment_starts_host,
    DefaultDeviceAllocator allocator,
    cudaStream_t stream)
{
    using cudautils::device_copy_n_async;
    const int32_t n_alignments           = get_size<int32_t>(alignments_out);
    const int64_t alignment_length_total = alignment_starts_host[n_alignments];
    thrust::transform(thrust::cuda::par(allocator).on(stream), alignment_sequences, alignment_sequences + alignment_length_total, alignment_sequences, translate_to_cigar_chars());
    thrust::reverse(thrust::cuda::par(allocator).on(stream), alignment_sequences, alignment_sequences + alignment_length_total);

    const int32_t max_length = std::abs(*std::max_element(alignment_lengths_host, alignment_lengths_host + n_alignments, [](int32_t a, int32_t b) { return std::abs(a) < std::abs(b); }));

    device_buffer<int> char_sequence_length(1, allocator, stream);
    device_buffer<int8_t> char_sequence(max_length, allocator, stream);
    device_buffer<int> char_counts(max_length, allocator, stream);

    size_t temp_storage_bytes = 0;
    for (int32_t i = 0; i < n_alignments; ++i)
    {
        size_t required_bytes = 0;
        cub::DeviceRunLengthEncode::Encode(nullptr, required_bytes, alignment_sequences, char_sequence.data(), char_counts.data(), char_sequence_length.data(), std::abs(alignment_lengths_host[i]), stream);
        temp_storage_bytes = std::max(temp_storage_bytes, required_bytes);
    }
    device_buffer<char> temp_storage(temp_storage_bytes, allocator, stream);

    {
        const int32_t alignment_length = std::abs(alignment_lengths_host[0]);
        cub::DeviceRunLengthEncode::Encode(temp_storage.data(), temp_storage_bytes, alignment_sequences + alignment_length_total - alignment_starts_host[0] - alignment_length, char_sequence.data(), char_counts.data(), char_sequence_length.data(), alignment_length, stream);
    }

    pinned_host_vector<char> cigar(max_length);
    pinned_host_vector<int32_t> cigar_counts(max_length);
    std::string cigar_string;
    for (int32_t i = 1; i <= n_alignments; ++i)
    {
        int cigar_length = 0;
        device_copy_n_async(char_sequence.data(), max_length, reinterpret_cast<int8_t*>(cigar.data()), stream);
        device_copy_n_async(char_counts.data(), max_length, cigar_counts.data(), stream);
        device_copy_n_async(char_sequence_length.data(), 1, &cigar_length, stream);
        GW_CU_CHECK_ERR(cudaStreamSynchronize(stream));
        if (i < n_alignments)
        {
            const int32_t alignment_length = std::abs(alignment_lengths_host[i]);
            cub::DeviceRunLengthEncode::Encode(temp_storage.data(), temp_storage_bytes, alignment_sequences + alignment_length_total - alignment_starts_host[i] - alignment_length, char_sequence.data(), char_counts.data(), char_sequence_length.data(), alignment_length, stream);
            thrust::transform(thrust::cuda::par(allocator).on(stream), char_sequence.data(), char_sequence.data() + alignment_length, char_sequence.data(), translate_to_cigar_chars());
        }

        cigar_string.clear();
        cigar_string.reserve(std::accumulate(begin(cigar_counts), begin(cigar_counts) + cigar_length, 0));
        for (int32_t j = 0; j < cigar_length; ++j)
        {
            cigar_string += std::to_string(cigar_counts[j]);
            cigar_string += cigar[j];
        }
        AlignmentImpl* alignment = dynamic_cast<AlignmentImpl*>(alignments_out[i - 1].get());
        const bool is_optimal    = (alignment_lengths_host[i - 1] >= 0);
        alignment->set_cigar(cigar_string.data(), get_size(cigar_string), is_optimal);
        alignment->set_status(alignment_lengths_host[i - 1] != 0 ? StatusType::success : StatusType::generic_error);
    }
}

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
