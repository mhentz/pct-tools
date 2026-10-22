#ifndef LIPSCHITZ_H
#define LIPSCHITZ_H

#include <string>
#include <vector>

#include <Eigen/Dense>

namespace lipschitz {

/// sum_i A_i^T (A_i x) for a system matrix split into row blocks A_i stored compressed on
/// disk. Blocks are read and multiplied in parallel; `x` must have as many entries as the
/// blocks have columns. This is the operation the power method needs to estimate the largest
/// eigenvalue of A^T A, i.e. the Lipschitz constant of the gradient of 1/2 ||A x - b||^2.
Eigen::VectorXf compute_ATAx(Eigen::VectorXf x, std::vector<std::string> const &matrix_filenames,
                             int n_threads = 1);

} // namespace lipschitz
#endif
