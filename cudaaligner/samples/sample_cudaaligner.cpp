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
#include <claragenomics/cudaaligner/aligner.hpp>
#include <claragenomics/cudaaligner/alignment.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/genomeutils.hpp>

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <random>
#include <fstream>

using namespace claragenomics;
using namespace claragenomics::genomeutils;
using namespace claragenomics::cudaaligner;

std::unique_ptr<Aligner> initialize_batch(int32_t max_query_size,
                                          int32_t max_target_size,
                                          int32_t max_alignments_per_batch,
                                          claragenomics::DefaultDeviceAllocator allocator)
{
    // Get device information.
    int32_t device_count = 0;
    CGA_CU_CHECK_ERR(cudaGetDeviceCount(&device_count));
    assert(device_count > 0);

    // Initialize internal logging framework.
    Init();

    // Initialize CUDA Aligner batch object for batched processing of alignments on the GPU.
    const int32_t device_id = 0;
    cudaStream_t stream     = 0;

    std::unique_ptr<Aligner> batch = create_aligner(max_query_size,
                                                    max_target_size,
                                                    max_alignments_per_batch,
                                                    AlignmentType::global_alignment,
                                                    allocator,
                                                    stream,
                                                    device_id);

    return std::move(batch);
}

void generate_data(std::vector<std::pair<std::string, std::string>>& data,
                   int32_t max_query_size,
                   int32_t max_target_size,
                   int32_t num_examples)
{
    std::minstd_rand rng(1);
    for (int32_t i = 0; i < num_examples; i++)
    {
        data.emplace_back(std::make_pair(
            generate_random_genome(max_query_size, rng),
            generate_random_genome(max_target_size, rng)));
    }
}

struct alignment_task
{
    std::string query;
    std::string target;
    bool query_reverse_strand;
    bool target_reverse_strand;
};

void throw_on_invalid_dna_sequence(const std::string& seq, const int32_t counter)
{
    const int32_t len = get_size<int32_t>(seq);
    for(int32_t c = 0; c < len; ++c)
    {
        const char x = seq[c];
        if(x != 'A' && x != 'C' && x != 'G' && x != 'T' && x != '\n')
        {
            throw std::runtime_error(std::string("Not a valid sequence: input file entry:") + std::to_string(counter) + " in char: " + std::to_string(c) + ", seq: " + seq);
        }
    }
}

std::vector<alignment_task> load_from_file(const std::string& filename, int32_t max_entries)
{
    std::vector<alignment_task> alignments;
    std::ifstream infile(filename);

    int32_t counter = 0;
    try{
        while(infile.good())
        {
                alignment_task x;
                char fwd = 0;
                infile >> fwd;
                x.query_reverse_strand = (fwd == 'R');
                infile >> x.query;
                throw_on_invalid_dna_sequence(x.query, counter);
                infile >> fwd;
                x.target_reverse_strand = (fwd == 'R');
                infile >> x.target;
                throw_on_invalid_dna_sequence(x.target, counter);
                infile >> fwd;
                if(fwd != '\n')
                    std::runtime_error("Expected a newline in file.");
                alignments.push_back(std::move(x));
            if(get_size(alignments) >= max_entries)
                break;
            ++counter;
        }
    } catch (...)
    {
    }

    return alignments;
}

void perform_alignments(Aligner* batch, const bool print)
{
    // Launch alignment on the GPU. align_all is an async call.
    batch->align_all();
    // Synchronize all alignments.
    batch->sync_alignments();
    if (print)
    {
        const std::vector<std::shared_ptr<Alignment>>& alignments = batch->get_alignments();
        for (const auto& alignment : alignments)
        {
            FormattedAlignment formatted = alignment->format_alignment();
            std::cout << formatted;
        }
    }
    // Reset batch to reuse memory for new alignments.
    batch->reset();
}

int main(int argc, char** argv)
{
    // Process options
    int c      = 0;
    bool help  = false;
    bool print = false;

    while ((c = getopt(argc, argv, "hp")) != -1)
    {
        switch (c)
        {
        case 'p':
            print = true;
            break;
        case 'h':
            help = true;
            break;
        }
    }

    if (help)
    {
        std::cout << "CUDA Aligner API sample program. Runs pairwise alignment over a batch of randomly generated sequences." << std::endl;
        std::cout << "Usage:" << std::endl;
        std::cout << "./sample_cudaaligner [-p] [-h]" << std::endl;
        std::cout << "-p : Print the MSA or consensus output to stdout" << std::endl;
        std::cout << "-h : Print help message" << std::endl;
        std::exit(0);
    }

    const int32_t num_entries  = 100000;
    const int32_t batch_size   = 20000;
    std::vector<alignment_task> data = load_from_file("/archive/ahehn/genomics/alignments_dump.dat", num_entries);
    
    int32_t max_query_length  = 0;
    int32_t max_target_length = 0;
    std::vector<int32_t> query_lengths;
    std::vector<int32_t> target_lengths;
    for(auto const& entry : data)
    {
        max_query_length = std::max(max_query_length, get_size<int32_t>(entry.query));
        max_target_length = std::max(max_target_length, get_size<int32_t>(entry.target));
        query_lengths.push_back(get_size<int32_t>(entry.query));
        target_lengths.push_back(get_size<int32_t>(entry.target));
    }
    std::cout << "max query length: " << max_query_length << "\t max target length: " << max_target_length << std::endl;
    sort(begin(query_lengths),end(query_lengths));
    sort(begin(target_lengths),end(target_lengths));
    std::cout << "query length percentiles: 10%: " << query_lengths[get_size(query_lengths)/10] << "\t 50%: " << query_lengths[get_size(query_lengths)/2] << "\t 90%: " << query_lengths[9*get_size(query_lengths)/10] << std::endl;
    std::cout << "target length percentiles: 10%: " << target_lengths[get_size(target_lengths)/10] << "\t 50%: " << target_lengths[get_size(target_lengths)/2] << "\t 90%: " << target_lengths[9*get_size(target_lengths)/10] << std::endl;

    const std::size_t max_gpu_memory                = claragenomics::cudautils::find_largest_contiguous_device_memory_section();
    claragenomics::DefaultDeviceAllocator allocator = create_default_device_allocator(max_gpu_memory);


    // Initialize batch.
    std::unique_ptr<Aligner> batch = initialize_batch(max_query_length, max_target_length, batch_size, allocator);

    // Generate data.
//    std::vector<std::pair<std::string, std::string>> data;
//    generate_data(data, query_length, target_length, num_entries);
    std::cout << "Running pairwise alignment for " << get_size(data) << " pairs..." << std::endl;

    // Loop over all the alignment pairs, add them to the batch and process them.
    int32_t counter = 0;
    for(auto const& entry : data)
    {
        // Add a pair to the batch, and check for status.
        StatusType status = batch->add_alignment(entry.query.c_str(), entry.query.length(), entry.target.c_str(), entry.target.length());
        if (status == exceeded_max_alignments)
        {
            perform_alignments(batch.get(), print);
            std::cout << "Aligned till " << counter << "." << std::endl;
            status = batch->add_alignment(entry.query.c_str(), entry.query.length(), entry.target.c_str(), entry.target.length());
        }
        if (status != success)
        {
            throw std::runtime_error("Experienced error type " + std::to_string(status));
        }
        ++counter;
    }
    perform_alignments(batch.get(), print);
    std::cout << "Aligned till " << counter << "." << std::endl;

    return 0;
}
