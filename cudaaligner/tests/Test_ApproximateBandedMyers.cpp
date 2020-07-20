/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "../src/aligner_global_myers_banded.hpp"
#include <claragenomics/utils/signed_integer_utils.hpp>
#include <claragenomics/cudaaligner/alignment.hpp>

#include <gtest/gtest.h>
#include <algorithm>
#include <limits>

namespace
{

constexpr int32_t word_size = 32;
int32_t get_max_sequence_length(std::vector<std::pair<std::string, std::string>> const& inputs)
{
    using claragenomics::get_size;
    int64_t max_string_size = 0;
    for (auto const& pair : inputs)
    {
        max_string_size = std::max(max_string_size, get_size(pair.first));
        max_string_size = std::max(max_string_size, get_size(pair.second));
    }
    return static_cast<int32_t>(max_string_size);
}

struct TestCase
{
    std::string query;
    std::string target;
};

std::vector<TestCase> create_band_test_cases()
{
    std::vector<TestCase> data;
    data.push_back({"AGGGCGAATATCGCCTCCCGCATTAAGCTGTACCTTCCAGCCCCGCCGGTAATTCCAGCCGGTTGAAGCCACGTCTGCCACGGCACAATGTTTTCGCTTTGCCCGGTGACGGATTTAATCCACCACAG", "AGGGCGAATATCGCCTCCGCATTAAACTGTACTTCCCAGCCCCGCCAGTATTCCAGCGGGTTGAAGCCGCGTCTGCCACAGCGCAATGTTTTCTTTGCCCACGGTGACCGGTTTAGTCACTACAGTTGC"});
    return data;
}

class TestApproximateBandedMyers : public ::testing::TestWithParam<TestCase>
{
};

} // namespace

TEST_P(TestApproximateBandedMyers, EditDistanceGrowsWithBand)
{
    using namespace claragenomics::cudaaligner;
    using namespace claragenomics;

    TestCase t = GetParam();

    DefaultDeviceAllocator allocator = create_default_device_allocator();

    const int32_t max_string_size = std::max(get_size<int32_t>(t.query), get_size<int32_t>(t.target));

    int32_t last_edit_distance      = std::numeric_limits<int32_t>::max();
    int32_t last_bw                 = -1;
    std::vector<int32_t> bandwidths = {2, 4, 16, 31, 32, 34, 63, 64, 66, 255, 256, 258, 1023, 1024, 1026, 2048};
    for (const int32_t max_bw : bandwidths)
    {
        if (max_bw % word_size == 1)
            continue; // not supported
        std::unique_ptr<Aligner> aligner = std::make_unique<AlignerGlobalMyersBanded>(-1,
                                                                                      max_bw,
                                                                                      allocator,
                                                                                      nullptr,
                                                                                      0);

        ASSERT_EQ(StatusType::success, aligner->add_alignment(t.query.c_str(), t.query.length(),
                                                              t.target.c_str(), t.target.length()))
            << "Could not add alignment to aligner";
        aligner->align_all();
        aligner->sync_alignments();
        const std::vector<std::shared_ptr<Alignment>>& alignments = aligner->get_alignments();
        ASSERT_EQ(get_size(alignments), 1);
        std::vector<AlignmentState> operations = alignments[0]->get_alignment();
        if (!operations.empty())
        {
            int32_t edit_distance = std::count_if(begin(operations), end(operations), [](AlignmentState x) { return x != AlignmentState::match; });
            ASSERT_LE(edit_distance, last_edit_distance) << "for max bandwidth = " << max_bw << " vs. max bandwidth = " << last_bw;
            last_edit_distance = edit_distance;
            last_bw            = max_bw;
        }
    }
};

INSTANTIATE_TEST_SUITE_P(TestApproximateBandedMyersInstance, TestApproximateBandedMyers, ::testing::ValuesIn(create_band_test_cases()));
