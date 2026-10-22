#ifndef RESIDUALS_H
#define RESIDUALS_H

#include <string>
#include <vector>

#include <Eigen/Dense>

namespace recon {

/// sum_i ||A_i x - b_i||^2 for a system split into blocks (A_i, b_i) stored on disk.
float squared_residuals(Eigen::VectorXf const &x, std::vector<std::string> const &matrix_filenames,
                        std::vector<std::string> const &b_filenames, int verbosity_level = 0,
                        int n_threads = 1);

} // namespace recon
#endif
