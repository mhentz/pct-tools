#ifndef IO_H
#define IO_H

#include <string>
#include <vector>

#include "blosc2.h"
#include <Eigen/Sparse>

using SparseMatrix = Eigen::SparseMatrix<float, Eigen::RowMajor, int>;

namespace io {

/// Default Blosc codec used when storing compressed matrices and vectors.
inline constexpr char const *DEFAULT_CODEC = "lz4hc";

void store_matrix(SparseMatrix &m, std::string const &filename);
void store_vector(Eigen::VectorXf const &vec, std::string const &filename);

void store_compressed_matrix(SparseMatrix &m, std::string const &filename, int compression_level = 9,
                             int shuffle_method = BLOSC_SHUFFLE, std::string const &codec = DEFAULT_CODEC,
                             int verbosity_level = 0);

void store_compressed_vector(Eigen::VectorXf const &vec, std::string const &filename,
                             int compression_level = 9, int shuffle_method = BLOSC_SHUFFLE,
                             std::string const &codec = DEFAULT_CODEC);

SparseMatrix read_matrix(std::string const &filename);
SparseMatrix read_compressed_matrix(std::string const &filename, int verbosity_level = 0);

Eigen::VectorXf read_vector(std::string const &filename);
Eigen::VectorXf read_compressed_vector(std::string const &filename);

void construct_matrix(std::string const &filename, std::vector<int> const &rows, std::vector<int> const &cols,
                      std::vector<float> const &values, int n_rows, int n_cols, int compression_level = 9,
                      int shuffle_method = BLOSC_SHUFFLE, std::string const &codec = DEFAULT_CODEC,
                      int verbosity_level = 0, int n_threads = 4);

} // namespace io
#endif
