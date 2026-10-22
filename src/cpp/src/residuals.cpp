#include "residuals.h"
#include "io.h"
#include "parallel.h"

#include <iostream>
#include <stdexcept>

#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

float recon::squared_residuals(Eigen::VectorXf const &x, std::vector<std::string> const &matrix_filenames,
                               std::vector<std::string> const &b_filenames, int const verbosity_level,
                               int const n_threads) {
    if (matrix_filenames.size() != b_filenames.size()) {
        throw std::invalid_argument("Number of matrices and b vectors must match!");
    }
    if (matrix_filenames.empty()) {
        throw std::invalid_argument("At least one matrix is required");
    }
    if (matrix_filenames[0] == b_filenames[0]) {
        throw std::invalid_argument("The matrix and b vector file names shouldn't match!");
    }

    omp_set_num_threads(n_threads);

    int const n_loops = static_cast<int>(matrix_filenames.size());

    if (verbosity_level > 0) {
        std::cout << "Compute squared residuals over " << n_loops << " blocks using " << n_threads
                  << " threads" << std::endl;
    }

    ExceptionCollector errors;

    std::vector<Eigen::VectorXf> b_vectors(n_loops);
#pragma omp parallel for
    for (int i = 0; i < n_loops; i++) {
        errors.run([&] { b_vectors[i] = io::read_vector(b_filenames[i]); });
    }
    errors.rethrow();

    std::vector<float> partial_sums(n_loops, 0.0f);

#pragma omp parallel for
    for (int i = 0; i < n_loops; i++) {
        errors.run([&] {
            SparseMatrix const A = io::read_compressed_matrix(matrix_filenames[i]);
            if (A.cols() != x.size()) {
                throw std::runtime_error("Matrix " + matrix_filenames[i] + " has " +
                                         std::to_string(A.cols()) + " columns but x has " +
                                         std::to_string(x.size()) + " elements");
            }
            if (A.rows() != b_vectors[i].size()) {
                throw std::runtime_error("Matrix " + matrix_filenames[i] + " has " +
                                         std::to_string(A.rows()) + " rows but b has " +
                                         std::to_string(b_vectors[i].size()) + " elements");
            }

            Eigen::VectorXf const residual = A * x - b_vectors[i];
            partial_sums[i] = residual.squaredNorm();
        });
    }
    errors.rethrow();

    // Summed in double: with thousands of blocks the float sum would lose digits.
    double res_sq = 0.0;
    for (float const partial : partial_sums) {
        res_sq += partial;
    }
    return static_cast<float>(res_sq);
}
