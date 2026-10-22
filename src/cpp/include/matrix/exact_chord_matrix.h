#ifndef EXACT_CHORD_MATRIX_H
#define EXACT_CHORD_MATRIX_H

#include <string>
#include <tuple>
#include <vector>

#include "blosc2.h"
#include "io.h"
#include <Eigen/Sparse>

using Triplet = Eigen::Triplet<float>;

namespace matrix {

/// Chord lengths of one trajectory (x, y, z) through every voxel of the image, as a dense
/// i x j x k array. Mainly used for testing and visualisation.
std::vector<std::vector<std::vector<float>>>
compute_matrix_elements(std::vector<float> const &x, std::vector<float> const &y, std::vector<float> const &z,
                        std::tuple<int, int, int> img_shape, std::tuple<float, float, float> coord_origin,
                        float pixel_width, float slice_thickness, bool is_verbose = false, int n_threads = 1);

/// Non-owning view of a block of trajectories: one pointer per trajectory and axis to
/// `n_points` contiguous coordinates. The caller keeps the coordinate storage alive. Built
/// from nested vectors by `make_block()` and from NumPy buffers by the Python bindings, so that
/// neither copies the coordinates.
struct TrajectoryBlock {
    std::vector<float const *> x, y, z;
    int n_points = 0;

    [[nodiscard]] int size() const { return static_cast<int>(x.size()); }
};

/// View of trajectories given as one coordinate vector per trajectory. Throws
/// std::invalid_argument if the block is empty or the vectors differ in length.
TrajectoryBlock make_block(std::vector<std::vector<float>> const &x_array,
                           std::vector<std::vector<float>> const &y_array,
                           std::vector<std::vector<float>> const &z_array);

/// Non-zero system-matrix entries (row = trajectory index, col = flat voxel index, value = chord
/// length) for a block of trajectories.
std::vector<Triplet> compute_matrix_triplets(TrajectoryBlock const &block,
                                             std::tuple<int, int, int> img_shape,
                                             std::tuple<float, float, float> coord_origin, float pixel_width,
                                             float slice_thickness, int verbosity_level = 0,
                                             int n_threads = 1);

/// Same for a block given as one coordinate vector per trajectory, see make_block().
std::vector<Triplet> compute_matrix_triplets(std::vector<std::vector<float>> const &x_array,
                                             std::vector<std::vector<float>> const &y_array,
                                             std::vector<std::vector<float>> const &z_array,
                                             std::tuple<int, int, int> img_shape,
                                             std::tuple<float, float, float> coord_origin, float pixel_width,
                                             float slice_thickness, int verbosity_level = 0,
                                             int n_threads = 1);

/// compute_matrix_triplets() followed by assembling the sparse matrix and storing it compressed.
void compute_and_store_sparse_matrix(std::string const &filename, TrajectoryBlock const &block,
                                     std::tuple<int, int, int> img_shape,
                                     std::tuple<float, float, float> coord_origin, float pixel_width,
                                     float slice_thickness, int verbosity_level = 0, int n_threads = 1,
                                     int compression_level = 9, int shuffle_method = BLOSC_SHUFFLE,
                                     std::string const &codec = io::DEFAULT_CODEC);

/// Same for a block given as one coordinate vector per trajectory, see make_block().
void compute_and_store_sparse_matrix(
    std::string const &filename, std::vector<std::vector<float>> const &x_array,
    std::vector<std::vector<float>> const &y_array, std::vector<std::vector<float>> const &z_array,
    std::tuple<int, int, int> img_shape, std::tuple<float, float, float> coord_origin, float pixel_width,
    float slice_thickness, int verbosity_level = 0, int n_threads = 1, int compression_level = 9,
    int shuffle_method = BLOSC_SHUFFLE, std::string const &codec = io::DEFAULT_CODEC);

} // namespace matrix

#endif /* EXACT_CHORD_MATRIX_H */
