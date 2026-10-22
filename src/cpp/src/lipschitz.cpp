#include "lipschitz.h"
#include "io.h"
#include "parallel.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

Eigen::VectorXf lipschitz::compute_ATAx(Eigen::VectorXf const x, std::vector<std::string> const &filenames,
                                        int const n_threads) {
    omp_set_num_threads(n_threads);

    int const n_cols = static_cast<int>(x.size());
    int const n_loops = static_cast<int>(filenames.size());

    ExceptionCollector errors;

    Eigen::VectorXf ATAx = Eigen::VectorXf::Zero(n_cols);

    // Each thread sums the products of its blocks into a private vector, merged into `ATAx`
    // one thread at a time afterwards. This needs one image per thread rather than one per
    // block, which matters with thousands of blocks and millions of voxels.
#pragma omp parallel
    {
        Eigen::VectorXf local = Eigen::VectorXf::Zero(n_cols);

#pragma omp for
        for (int i = 0; i < n_loops; i++) {
            errors.run([&] {
                SparseMatrix const A = io::read_compressed_matrix(filenames[i]);
                if (A.cols() != x.size()) {
                    throw std::runtime_error("Matrix " + filenames[i] + " has " + std::to_string(A.cols()) +
                                             " columns but x has " + std::to_string(x.size()) + " elements");
                }
                local += A.transpose() * (A * x);
            });
        }

#pragma omp critical(pct_tools_lipschitz_reduce)
        ATAx += local;
    }
    errors.rethrow();

    return ATAx;
}
