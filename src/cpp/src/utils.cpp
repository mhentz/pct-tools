#include "utils.h"
#include "io.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Sparse>

void utils::recompress_matrix(std::string const &filename, std::string const &output,
                              int const compression_level, int const shuffle_method, std::string const &codec,
                              int const verbosity_level) {
    SparseMatrix mat = io::read_compressed_matrix(filename);
    io::store_compressed_matrix(mat, output, compression_level, shuffle_method, codec, verbosity_level);
}

void utils::subsample_matrix_vector_pair(std::string const &input_path, std::string const &projection_angle,
                                         std::string const &output_path, float const ratio,
                                         std::string const &pixel_w_str, unsigned seed,
                                         int const compression_level, int const shuffle_method,
                                         std::string const &codec, int const verbosity_level) {
    if (input_path == output_path) {
        throw std::invalid_argument("Input and output paths shouldn't match.");
    }
    if (ratio < 0.0f || ratio > 1.0f) {
        throw std::invalid_argument("The subsampling ratio must lie in [0, 1].");
    }

    std::string const matrix_filename =
        input_path + "/matrix-" + pixel_w_str + "/" + projection_angle + ".mat";
    std::string const vector_filename = input_path + "/b/b_" + projection_angle + ".vec";

    if (verbosity_level > 0) {
        std::cout << "reading matrix " << matrix_filename << '\n';
        std::cout << "reading vector " << vector_filename << std::endl;
    }

    SparseMatrix const mat = io::read_compressed_matrix(matrix_filename);
    Eigen::VectorXf const b = io::read_vector(vector_filename);
    if (mat.rows() != b.size()) {
        throw std::runtime_error("Matrix has " + std::to_string(mat.rows()) + " rows but b has " +
                                 std::to_string(b.size()) + " elements");
    }

    // Sample row indices
    std::vector<int> indices(mat.rows());
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    int const n_samples = static_cast<int>(std::floor(ratio * static_cast<float>(mat.rows())));
    std::vector<int> const samples(indices.begin(), indices.begin() + n_samples);

    // Construct subsampled matrix
    SparseMatrix subsampled_mat(n_samples, mat.cols());
    for (int k = 0; k < n_samples; ++k)
        for (SparseMatrix::InnerIterator it(mat, samples[k]); it; ++it)
            subsampled_mat.insert(k, it.col()) = it.value();
    subsampled_mat.makeCompressed();

    // Construct subsampled vector
    Eigen::VectorXf subsampled_vec(n_samples);
    for (int i = 0; i < n_samples; i++)
        subsampled_vec(i) = b(samples[i]);

    // Store matrix and b-vector
    io::store_compressed_matrix(subsampled_mat,
                                output_path + "/matrix-" + pixel_w_str + "/" + projection_angle + ".mat",
                                compression_level, shuffle_method, codec, verbosity_level);
    io::store_vector(subsampled_vec, output_path + "/b/b_" + projection_angle + ".vec");
}
