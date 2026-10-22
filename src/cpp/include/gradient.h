#ifndef GRADIENT_H
#define GRADIENT_H

#include <string>
#include <vector>

#include <Eigen/Dense>

namespace recon {

/// One step of steepest descent on 1/2 ||A x - b||^2 for a system split into blocks
/// (A_i, b_i) stored on disk: returns x - step_size * sum_i A_i^T (A_i x - b_i).
Eigen::VectorXf gradient_descent(Eigen::VectorXf x, float step_size,
                                 std::vector<std::string> const &matrix_filenames,
                                 std::vector<std::string> const &b_filenames, int verbosity_level = 0,
                                 int n_threads = 1);

} // namespace recon
#endif
