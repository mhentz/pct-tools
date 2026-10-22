#include "exact_chord_matrix.h"
#include "gradient.h"
#include "io.h"
#include "lipschitz.h"
#include "mlp.h"
#include "residuals.h"
#include "utils.h"

#include <stdexcept>
#include <string>
#include <tuple>

#include <Eigen/Sparse>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using Triplet = Eigen::Triplet<float>;
namespace py = pybind11;

/// A 2-D float32 array accepted without copying when it is already C-contiguous float32;
/// other dtypes and layouts are converted once on entry.
using CoordinateArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {

/// Non-owning view of three (n_trajectories, n_samples) coordinate arrays. The arrays must
/// outlive the view; the callers below hold them for the duration of the call.
matrix::TrajectoryBlock block_view(CoordinateArray const &x, CoordinateArray const &y,
                                   CoordinateArray const &z) {
    if (x.ndim() != 2 || y.ndim() != 2 || z.ndim() != 2) {
        throw std::invalid_argument("input x_array must have 2 dimensions");
    }
    if (x.shape(0) != y.shape(0) || x.shape(0) != z.shape(0)) {
        throw std::invalid_argument("input arrays must have the same number of trajectories");
    }
    if (x.shape(1) != y.shape(1) || x.shape(1) != z.shape(1)) {
        throw std::invalid_argument("input arrays must have the same size");
    }
    if (x.shape(0) == 0) {
        throw std::invalid_argument("input arrays must contain at least one trajectory");
    }
    if (x.shape(1) == 0) {
        throw std::invalid_argument("input x_array must have 2 dimensions");
    }

    auto const n_traj = static_cast<size_t>(x.shape(0));
    auto const n_points = static_cast<size_t>(x.shape(1));

    matrix::TrajectoryBlock block;
    block.n_points = static_cast<int>(n_points);
    block.x.resize(n_traj);
    block.y.resize(n_traj);
    block.z.resize(n_traj);
    for (size_t b = 0; b < n_traj; b++) {
        block.x[b] = x.data() + (b * n_points);
        block.y[b] = y.data() + (b * n_points);
        block.z[b] = z.data() + (b * n_points);
    }
    return block;
}

} // namespace

namespace pybind11 {
namespace detail {

template <> struct type_caster<Triplet> {
  public:
    PYBIND11_TYPE_CASTER(Triplet, _("Triplet"));

    // Python to C++: not needed, triplets only flow out of the extension.
    bool load(handle, bool) { return false; }

    // C++ to Python: (row, col, value)
    static handle cast(Triplet src, return_value_policy, handle) {
        return py::make_tuple(src.row(), src.col(), src.value()).release();
    }
};

template <> struct type_caster<std::vector<std::vector<std::vector<float>>>> {
  public:
    PYBIND11_TYPE_CASTER(std::vector<std::vector<std::vector<float>>>,
                         _("std::vector<std::vector<std::vector<float>>>"));

    bool load(handle, bool) { return false; }

    // C++ to Python: a 3D NumPy array
    static handle cast(std::vector<std::vector<std::vector<float>>> src, return_value_policy, handle) {
        if (src.empty() || src[0].empty() || src[0][0].empty()) {
            throw std::runtime_error("3D vector is empty or not properly formed");
        }

        size_t const depth = src.size();
        size_t const rows = src[0].size();
        size_t const cols = src[0][0].size();

        auto result = py::array_t<float>({depth, rows, cols});
        auto r = result.mutable_unchecked<3>();

        for (size_t i = 0; i < depth; ++i) {
            for (size_t j = 0; j < rows; ++j) {
                for (size_t k = 0; k < cols; ++k) {
                    r(i, j, k) = src[i][j][k];
                }
            }
        }
        return result.release();
    }
};

} // namespace detail
} // namespace pybind11

PYBIND11_MODULE(_ext, m) {
    m.doc() = "C++ kernels of pct-tools: sparse matrix I/O, system-matrix construction, most likely path "
              "and reconstruction steps.";

    // ---- I/O -------------------------------------------------------------------------------
    m.def("store_vector", &io::store_vector, "Store a vector.", py::arg("vector"), py::arg("filename"));
    m.def("store_compressed_vector", &io::store_compressed_vector, "Compress and store a vector.",
          py::arg("vector"), py::arg("filename"), py::arg("compression_level") = 9,
          py::arg("shuffle_method") = static_cast<int>(BLOSC_SHUFFLE), py::arg("codec") = io::DEFAULT_CODEC);
    m.def("read_vector", &io::read_vector, "Read a vector.", py::arg("filename"));
    m.def("read_compressed_vector", &io::read_compressed_vector, "Read a compressed vector.",
          py::arg("filename"));

    m.def("read_matrix", &io::read_matrix, "Read an uncompressed matrix.", py::arg("filename"));
    m.def("read_compressed_matrix", &io::read_compressed_matrix, "Read a compressed matrix.",
          py::arg("filename"), py::arg("verbosity_level") = 0);
    m.def("store_compressed_matrix", &io::store_compressed_matrix, "Compress and store a matrix.",
          py::arg("matrix"), py::arg("filename"), py::arg("compression_level") = 9,
          py::arg("shuffle_method") = static_cast<int>(BLOSC_SHUFFLE), py::arg("codec") = io::DEFAULT_CODEC,
          py::arg("verbosity_level") = 0);
    m.def("construct_matrix", &io::construct_matrix,
          "Construct a sparse matrix from row indices, column indices and values and store it compressed.",
          py::arg("filename"), py::arg("rows"), py::arg("cols"), py::arg("values"), py::arg("n_rows"),
          py::arg("n_cols"), py::arg("compression_level") = 9,
          py::arg("shuffle_method") = static_cast<int>(BLOSC_SHUFFLE), py::arg("codec") = io::DEFAULT_CODEC,
          py::arg("verbosity_level") = 0, py::arg("n_threads") = 4);

    // ---- Lipschitz constant ----------------------------------------------------------------
    m.def("compute_ATAx", &lipschitz::compute_ATAx, "Compute A^T A x over a set of matrix files.",
          py::arg("x"), py::arg("filenames"), py::arg("n_threads") = 1);

    // ---- System matrix ---------------------------------------------------------------------
    m.def("compute_matrix_elements", &matrix::compute_matrix_elements,
          "Compute the dense chord-length array for a single proton trajectory.", py::arg("x"), py::arg("y"),
          py::arg("z"), py::arg("img_shape"), py::arg("coord_origin"), py::arg("pixel_width"),
          py::arg("slice_thickness"), py::arg("is_verbose") = false, py::arg("n_threads") = 1);
    // The coordinate arrays are passed as NumPy buffers and read in place rather than
    // converted to nested vectors, which would copy every sample through a Python object.
    m.def(
        "compute_matrix_triplets",
        [](CoordinateArray const &x, CoordinateArray const &y, CoordinateArray const &z,
           std::tuple<int, int, int> img_shape, std::tuple<float, float, float> coord_origin,
           float pixel_width, float slice_thickness, int verbosity_level, int n_threads) {
            return matrix::compute_matrix_triplets(block_view(x, y, z), img_shape, coord_origin, pixel_width,
                                                   slice_thickness, verbosity_level, n_threads);
        },
        "Compute the (row, col, value) triplets of the sparse system matrix for a block of trajectories.",
        py::arg("x"), py::arg("y"), py::arg("z"), py::arg("img_shape"), py::arg("coord_origin"),
        py::arg("pixel_width"), py::arg("slice_thickness"), py::arg("verbosity_level") = 0,
        py::arg("n_threads") = 1);
    m.def(
        "compute_and_store_sparse_matrix",
        [](std::string const &filename, CoordinateArray const &x, CoordinateArray const &y,
           CoordinateArray const &z, std::tuple<int, int, int> img_shape,
           std::tuple<float, float, float> coord_origin, float pixel_width, float slice_thickness,
           int verbosity_level, int n_threads, int compression_level, int shuffle_method,
           std::string const &codec) {
            matrix::compute_and_store_sparse_matrix(filename, block_view(x, y, z), img_shape, coord_origin,
                                                    pixel_width, slice_thickness, verbosity_level, n_threads,
                                                    compression_level, shuffle_method, codec);
        },
        "Compute the sparse system matrix for a block of trajectories and store it compressed.",
        py::arg("filename"), py::arg("x"), py::arg("y"), py::arg("z"), py::arg("img_shape"),
        py::arg("coord_origin"), py::arg("pixel_width"), py::arg("slice_thickness"),
        py::arg("verbosity_level") = 0, py::arg("n_threads") = 1, py::arg("compression_level") = 9,
        py::arg("shuffle_method") = static_cast<int>(BLOSC_SHUFFLE), py::arg("codec") = io::DEFAULT_CODEC);

    // ---- Most likely path ------------------------------------------------------------------
    m.def("get_coefficients", &mlp::get_coefficients,
          "Built-in scattering-model coefficients for a named phantom.", py::arg("phantom_type"));
    m.def("compute_mlp", &mlp::compute_mlp, "Compute the most likely path for polynomial coefficients.",
          py::arg("z"), py::arg("h_in"), py::arg("theta_in"), py::arg("h_out"), py::arg("theta_out"),
          py::arg("coefficients"), py::arg("n_threads") = 1);

    // ---- Reconstruction --------------------------------------------------------------------
    m.def("gradient_descent", &recon::gradient_descent, "Perform one steepest-descent step.", py::arg("x"),
          py::arg("step_size"), py::arg("matrix_filenames"), py::arg("b_filenames"),
          py::arg("verbosity_level") = 0, py::arg("n_threads") = 1);
    m.def("squared_residuals", &recon::squared_residuals, "Compute the squared residual norm ||Ax - b||^2.",
          py::arg("x"), py::arg("matrix_filenames"), py::arg("b_filenames"), py::arg("verbosity_level") = 0,
          py::arg("n_threads") = 1);

    // ---- Utilities -------------------------------------------------------------------------
    m.def("recompress_matrix", &utils::recompress_matrix,
          "Load a matrix and store it with a given codec and compression level.", py::arg("filename"),
          py::arg("output"), py::arg("compression_level") = 9,
          py::arg("shuffle_method") = static_cast<int>(BLOSC_SHUFFLE), py::arg("codec") = io::DEFAULT_CODEC,
          py::arg("verbosity_level") = 0);
    m.def("subsample_matrix_vector_pair", &utils::subsample_matrix_vector_pair,
          "Subsample corresponding matrix and b-vector for a given projection angle.", py::arg("input_path"),
          py::arg("projection_angle"), py::arg("output_path"), py::arg("ratio"), py::arg("pixel_w_str"),
          py::arg("seed") = 0, py::arg("compression_level") = 9,
          py::arg("shuffle_method") = static_cast<int>(BLOSC_SHUFFLE), py::arg("codec") = io::DEFAULT_CODEC,
          py::arg("verbosity_level") = 0);
}
