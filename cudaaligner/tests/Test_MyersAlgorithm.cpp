/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <claragenomics/cudaaligner/cudaaligner.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>
#include <claragenomics/utils/mathutils.hpp>
#include "../src/myers_gpu.cuh"
#include "../src/needleman_wunsch_cpu.hpp"
#include "cudaaligner_test_cases.hpp"

#include <algorithm>

#include <gtest/gtest.h>

namespace claragenomics
{

namespace cudaaligner
{
namespace
{

int32_t popc(const myers::WordType x)
{
    static_assert(sizeof(myers::WordType) == 4);
    constexpr int32_t nbits[16] = {0, 1, 1, 2,
                                   1, 2, 2, 3,
                                   1, 2, 2, 3,
                                   2, 3, 3, 4};

    int32_t cnt = nbits[x & 0xf];
    cnt += nbits[(x >> 4) & 0xf];
    cnt += nbits[(x >> 8) & 0xf];
    cnt += nbits[(x >> 12) & 0xf];
    cnt += nbits[(x >> 16) & 0xf];
    cnt += nbits[(x >> 20) & 0xf];
    cnt += nbits[(x >> 24) & 0xf];
    cnt += nbits[(x >> 28) & 0xf];
    return cnt;
}

int32_t get_myers_score(const int32_t i, const int32_t j, matrix<myers::WordType> const& pv, matrix<myers::WordType> const& mv, matrix<int32_t> const& score, const myers::WordType last_entry_mask)
{
    assert(i > 0); // row 0 is implicit, NW matrix is shifted by i -> i-1
    const int32_t word_idx = (i - 1) / myers::word_size;
    const int32_t bit_idx  = (i - 1) % myers::word_size;
    int32_t s              = score(word_idx, j);
    myers::WordType mask   = (~myers::WordType(1)) << bit_idx;
    if (word_idx == score.num_rows() - 1)
        mask &= last_entry_mask;
    s -= popc(mask & pv(word_idx, j));
    s += popc(mask & mv(word_idx, j));
    return s;
}
} // namespace

class TestMyersEditDistance : public ::testing::TestWithParam<TestCaseData>
{
};

TEST_P(TestMyersEditDistance, TestCases)
{
    TestCaseData t = GetParam();

    int32_t d         = myers_compute_edit_distance(t.target, t.query);
    matrix<int32_t> r = needleman_wunsch_build_score_matrix_naive(t.target, t.query);
    int32_t reference = r(r.num_rows() - 1, r.num_cols() - 1);
    ASSERT_EQ(d, reference);
}

class TestMyersScoreMatrix : public ::testing::TestWithParam<TestCaseData>
{
};

TEST_P(TestMyersScoreMatrix, TestCases)
{
    TestCaseData t = GetParam();

    matrix<int32_t> m = myers_get_full_score_matrix(t.target, t.query);
    matrix<int32_t> r = needleman_wunsch_build_score_matrix_naive(t.target, t.query);

    ASSERT_EQ(m.num_rows(), r.num_rows());
    ASSERT_EQ(m.num_cols(), r.num_cols());

    for (int32_t j = 0; j < m.num_cols(); ++j)
    {
        for (int32_t i = 0; i < m.num_rows(); ++i)
        {
            EXPECT_EQ(m(i, j), r(i, j)) << "index: (" << i << "," << j << ")";
        }
    }
}

class TestMyersBandedMatrixDeltas : public ::testing::TestWithParam<TestCaseData>
{
};

TEST_P(TestMyersBandedMatrixDeltas, TestCases)
{
    // Test if adjacent matrix entries
    // do not differ by more than delta = +/-1.

    using cudautils::device_copy_n;
    using cudautils::set_device_value;
    using myers::word_size;
    using myers::WordType;
    TestCaseData t = GetParam();

    // Skip tests for which myers_banded_gpu is not defined
    if (get_size(t.query) == 0 || get_size(t.target) == 0)
        return;

    cudaStream_t stream;
    CGA_CU_CHECK_ERR(cudaStreamCreate(&stream));
    {
        DefaultDeviceAllocator allocator = create_default_device_allocator();

        const int32_t query_size  = get_size<int32_t>(t.query);
        const int32_t target_size = get_size<int32_t>(t.target);
        device_buffer<char> query_d(query_size, allocator, stream);
        device_buffer<char> target_d(target_size, allocator, stream);
        device_copy_n(t.query.c_str(), query_size, query_d.data(), stream);
        device_copy_n(t.target.c_str(), target_size, target_d.data(), stream);

        CGA_CU_CHECK_ERR(cudaStreamSynchronize(stream));

        const int32_t max_distance_estimate = std::max(target_size, query_size) / 4;

        int32_t p          = min3(target_size, query_size, (max_distance_estimate - abs(target_size - query_size)) / 2);
        int32_t band_width = std::min(1 + 2 * p + abs(target_size - query_size), query_size);
        if (band_width % word_size == 1 && band_width != query_size) // we need at least two bits in the last word
        {
            p += 1;
            band_width = std::min(1 + 2 * p + abs(target_size - query_size), query_size);
        }
        const int32_t n_words      = ceiling_divide(query_size, word_size);
        const int32_t n_words_band = ceiling_divide(band_width, word_size);

        batched_device_matrices<myers::WordType> pvs(1, n_words_band * (target_size + 1), allocator, stream);
        batched_device_matrices<myers::WordType> mvs(1, n_words_band * (target_size + 1), allocator, stream);
        batched_device_matrices<int32_t> scores(1, n_words_band * (target_size + 1), allocator, stream);
        batched_device_matrices<myers::WordType> query_patterns(1, n_words * 4, allocator, stream);

        detail::test::myers_compute_scores_edit_dist_banded_test_function(
            pvs, mvs, scores, query_patterns,
            target_d.data(), query_d.data(), target_size, query_size, band_width, p, stream);

        const int32_t n_rows             = n_words_band;
        const int32_t n_cols             = target_size + 1;
        const matrix<int32_t> score      = scores.get_matrix(0, n_rows, n_cols, stream);
        const matrix<myers::WordType> pv = pvs.get_matrix(0, n_rows, n_cols, stream);
        const matrix<myers::WordType> mv = mvs.get_matrix(0, n_rows, n_cols, stream);

        const WordType last_entry_mask = band_width % word_size != 0 ? (WordType(1) << (band_width % word_size)) - 1 : ~WordType(0);

        // Check consistency along rows
        int32_t last_first_col_score = 0;
        for (int32_t i = 1; i < band_width + 1; ++i)
        {
            int32_t last_score = last_first_col_score;
            for (int32_t j = 0; j < target_size + 1; ++j)
            {
                const int32_t this_score = get_myers_score(i, j, pv, mv, score, last_entry_mask);
                EXPECT_LE(std::abs(last_score - this_score), 1) << " error at (" << i << "," << j << ")";
                last_score = this_score;
                if (j == 0)
                {
                    last_first_col_score = this_score;
                }
            }
        }

        // Check consistency along cols
        int32_t last_first_row_score = 1;
        for (int32_t j = 0; j < target_size + 1; ++j)
        {
            int32_t last_score = last_first_row_score;
            for (int32_t i = 1; i < band_width + 1; ++i)
            {
                const int32_t this_score = get_myers_score(i, j, pv, mv, score, last_entry_mask);
                EXPECT_LE(std::abs(last_score - this_score), 1) << " error at (" << i << "," << j << ")";
                last_score = this_score;
                if (i == 1)
                {
                    last_first_row_score = this_score;
                }
            }
        }
    }
    CGA_CU_CHECK_ERR(cudaStreamDestroy(stream));
}

INSTANTIATE_TEST_SUITE_P(TestMyersAlgorithm, TestMyersEditDistance, ::testing::ValuesIn(create_cudaaligner_test_cases()));
INSTANTIATE_TEST_SUITE_P(TestMyersAlgorithm, TestMyersScoreMatrix, ::testing::ValuesIn(create_cudaaligner_test_cases()));
INSTANTIATE_TEST_SUITE_P(TestMyersAlgorithm, TestMyersBandedMatrixDeltas, ::testing::ValuesIn(create_cudaaligner_test_cases()));

} // namespace cudaaligner
} // namespace claragenomics
