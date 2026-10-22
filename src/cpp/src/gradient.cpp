#include "gradient.h"
#include "io.h"
#include "parallel.h"

#include <iostream>
#include <stdexcept>

#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

Eigen::VectorXf recon::gradient_descent(Eigen::VectorXf x, float const step_size,
                                        std::vector<std::string> const &matrix_filenames,
                                        std::vector<std::string> const &b_filenames,
                                        int const verbosity_level, int const n_threads) {
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

    int const n_cols = static_cast<int>(x.size());
    int const n_loops = static_cast<int>(matrix_filenames.size());

    if (verbosity_level > 0) {
        std::cout << "Compute gradient descent step of size " << step_size << " over " << n_loops
                  << " blocks using " << n_threads << " threads" << std::endl;
    }

    ExceptionCollector errors;

    std::vector<Eigen::VectorXf> b_vectors(n_loops);
#pragma omp parallel for
    for (int i = 0; i < n_loops; i++) {
        errors.run([&] { b_vectors[i] = io::read_vector(b_filenames[i]); });
    }
    errors.rethrow();

    Eigen::VectorXf grad = Eigen::VectorXf::Zero(n_cols);

    // Each thread sums the gradients of its blocks into a private vector; the private vectors
    // are then added to `grad` one thread at a time, so no two threads update it concurrently.
#pragma omp parallel
    {
        Eigen::VectorXf grad_local = Eigen::VectorXf::Zero(n_cols);

#pragma omp for
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

                grad_local += A.transpose() * (A * x - b_vectors[i]);
            });
        }

#pragma omp critical(pct_tools_gradient_reduce)
        grad += grad_local;
    }
    errors.rethrow();

    x -= step_size * grad;
    return x;
}
