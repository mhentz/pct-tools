#include "io.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <omp.h>

#include "blosc2.h"
#include <Eigen/Sparse>

using StorageIndex = SparseMatrix::StorageIndex;
using Triplet = Eigen::Triplet<float>;

namespace {

// ------------------------------------------------------------------------------------------
// Blosc helpers. Per-call contexts are used instead of the blosc1_* API, whose compressor
// setting is process-global and therefore not safe to change from several threads.
// ------------------------------------------------------------------------------------------

int compcode_from_name(std::string const &codec) {
    int const code = blosc2_compname_to_compcode(codec.c_str());
    if (code < 0) {
        throw std::invalid_argument("Unknown Blosc codec '" + codec + "'");
    }
    return code;
}

void check_shuffle_method(int shuffle_method) {
    if (shuffle_method != BLOSC_NOSHUFFLE && shuffle_method != BLOSC_SHUFFLE &&
        shuffle_method != BLOSC_BITSHUFFLE) {
        throw std::invalid_argument(
            "Shuffle method not supported. Pick BLOSC_NOSHUFFLE=" + std::to_string(BLOSC_NOSHUFFLE) +
            ", BLOSC_SHUFFLE=" + std::to_string(BLOSC_SHUFFLE) +
            " or BLOSC_BITSHUFFLE=" + std::to_string(BLOSC_BITSHUFFLE));
    }
}

int32_t checked_chunk_size(size_t n_bytes) {
    if (n_bytes > static_cast<size_t>(BLOSC2_MAX_BUFFERSIZE)) {
        throw std::length_error("Buffer of " + std::to_string(n_bytes) +
                                " bytes exceeds the maximum Blosc chunk size");
    }
    return static_cast<int32_t>(n_bytes);
}

std::vector<char> compress(void const *src, size_t src_bytes, int typesize, int compression_level,
                           int shuffle_method, int compcode) {
    blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
    cparams.compcode = static_cast<uint8_t>(compcode);
    cparams.clevel = static_cast<uint8_t>(compression_level);
    cparams.typesize = typesize;
    cparams.nthreads = 1;
    cparams.filters[BLOSC2_MAX_FILTERS - 1] = static_cast<uint8_t>(shuffle_method);

    std::vector<char> dst(src_bytes + BLOSC2_MAX_OVERHEAD);
    blosc2_context *ctx = blosc2_create_cctx(cparams);
    int const n = blosc2_compress_ctx(ctx, src, checked_chunk_size(src_bytes), dst.data(),
                                      checked_chunk_size(dst.size()));
    blosc2_free_ctx(ctx);

    if (n <= 0) {
        throw std::runtime_error("Blosc compression failed with error code " + std::to_string(n));
    }
    dst.resize(static_cast<size_t>(n));
    return dst;
}

void decompress(std::vector<char> const &src, void *dst, size_t dst_bytes) {
    blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
    dparams.nthreads = 1;

    blosc2_context *ctx = blosc2_create_dctx(dparams);
    int const n = blosc2_decompress_ctx(ctx, src.data(), checked_chunk_size(src.size()), dst,
                                        checked_chunk_size(dst_bytes));
    blosc2_free_ctx(ctx);

    if (n < 0 || static_cast<size_t>(n) != dst_bytes) {
        throw std::runtime_error("Blosc decompression failed with error code " + std::to_string(n));
    }
}

// ------------------------------------------------------------------------------------------
// Stream helpers that fail loudly on short reads/writes instead of returning garbage.
// ------------------------------------------------------------------------------------------

std::ofstream open_for_writing(std::string const &filename) {
    std::ofstream os(filename, std::ios::binary);
    if (!os.is_open()) {
        throw std::runtime_error("Could not open file " + filename + " for writing");
    }
    return os;
}

std::ifstream open_for_reading(std::string const &filename) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Could not open file " + filename + " for reading");
    }
    return is;
}

template <class T> void write_scalar(std::ofstream &os, T const &value) {
    os.write(reinterpret_cast<char const *>(&value), sizeof(T));
}

void write_bytes(std::ofstream &os, void const *data, size_t n_bytes) {
    os.write(reinterpret_cast<char const *>(data), static_cast<std::streamsize>(n_bytes));
}

void read_bytes(std::ifstream &is, void *data, size_t n_bytes, std::string const &what) {
    is.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(n_bytes));
    if (static_cast<size_t>(is.gcount()) != n_bytes) {
        throw std::runtime_error("Unexpected end of file while reading " + what);
    }
}

template <class T> T read_scalar(std::ifstream &is, std::string const &what) {
    T value;
    read_bytes(is, &value, sizeof(T), what);
    return value;
}

std::vector<char> read_blob(std::ifstream &is, int n_bytes, std::string const &what) {
    if (n_bytes < 0) {
        throw std::runtime_error("Negative size stored for " + what);
    }
    std::vector<char> blob(static_cast<size_t>(n_bytes));
    read_bytes(is, blob.data(), blob.size(), what);
    return blob;
}

void require_non_negative(StorageIndex value, std::string const &what) {
    if (value < 0) {
        throw std::runtime_error("Negative " + what + " stored in file");
    }
}

} // namespace

// ------------------------------------------------------------------------------------------
// Uncompressed formats
// ------------------------------------------------------------------------------------------

void io::store_matrix(SparseMatrix &m, std::string const &filename) {
    m.makeCompressed();
    std::ofstream os = open_for_writing(filename);

    StorageIndex const rows = m.rows();
    StorageIndex const cols = m.cols();
    StorageIndex const nnz = m.nonZeros();
    StorageIndex const outer_size = m.outerSize();

    write_scalar(os, rows);
    write_scalar(os, cols);
    write_scalar(os, nnz);
    write_scalar(os, outer_size);

    write_bytes(os, m.valuePtr(), sizeof(float) * nnz);
    write_bytes(os, m.outerIndexPtr(), sizeof(StorageIndex) * outer_size);
    write_bytes(os, m.innerIndexPtr(), sizeof(StorageIndex) * nnz);
}

void io::store_vector(Eigen::VectorXf const &vec, std::string const &filename) {
    std::ofstream os = open_for_writing(filename);

    int const rows = static_cast<int>(vec.size());
    write_scalar(os, rows);
    write_bytes(os, vec.data(), sizeof(float) * rows);
}

SparseMatrix io::read_matrix(std::string const &filename) {
    std::ifstream is = open_for_reading(filename);

    auto const rows = read_scalar<StorageIndex>(is, "rows");
    auto const cols = read_scalar<StorageIndex>(is, "cols");
    auto const nnz = read_scalar<StorageIndex>(is, "nnz");
    auto const outer_size = read_scalar<StorageIndex>(is, "outer size");
    require_non_negative(rows, "number of rows");
    require_non_negative(cols, "number of columns");
    require_non_negative(nnz, "number of non-zeros");
    require_non_negative(outer_size, "outer size");

    SparseMatrix mat(rows, cols);
    mat.makeCompressed();
    mat.resizeNonZeros(nnz);

    read_bytes(is, mat.valuePtr(), sizeof(float) * nnz, "values");
    read_bytes(is, mat.outerIndexPtr(), sizeof(StorageIndex) * outer_size, "outer indices");
    read_bytes(is, mat.innerIndexPtr(), sizeof(StorageIndex) * nnz, "inner indices");

    mat.finalize();
    return mat;
}

Eigen::VectorXf io::read_vector(std::string const &filename) {
    std::ifstream is = open_for_reading(filename);

    auto const rows = read_scalar<int>(is, "size");
    if (rows < 0) {
        throw std::runtime_error("Negative vector size stored in file " + filename);
    }

    Eigen::VectorXf vec(rows);
    read_bytes(is, vec.data(), sizeof(float) * rows, "values");
    return vec;
}

// ------------------------------------------------------------------------------------------
// Compressed formats. Layout: rows, cols, nnz (StorageIndex each); three compressed-chunk
// sizes (int each); then the Blosc chunks for values, outer indices and inner indices.
// ------------------------------------------------------------------------------------------

void io::store_compressed_matrix(SparseMatrix &m, std::string const &filename, int compression_level,
                                 int shuffle_method, std::string const &codec, int verbosity_level) {
    check_shuffle_method(shuffle_method);
    int const compcode = compcode_from_name(codec);

    if (verbosity_level > 0) {
        std::cout << "Store matrix " << filename << " with codec " << codec << " at compression level "
                  << compression_level << std::endl;
    }

    m.makeCompressed();
    StorageIndex const rows = m.rows();
    StorageIndex const cols = m.cols();
    StorageIndex const nnz = m.nonZeros();

    size_t const values_bytes = sizeof(float) * nnz;
    size_t const oindex_bytes = sizeof(StorageIndex) * (rows + 1);
    size_t const iindex_bytes = sizeof(StorageIndex) * nnz;

    std::vector<char> const comp_values =
        compress(m.valuePtr(), values_bytes, sizeof(float), compression_level, shuffle_method, compcode);
    std::vector<char> const comp_oindex = compress(m.outerIndexPtr(), oindex_bytes, sizeof(StorageIndex),
                                                   compression_level, shuffle_method, compcode);
    std::vector<char> const comp_iindex = compress(m.innerIndexPtr(), iindex_bytes, sizeof(StorageIndex),
                                                   compression_level, shuffle_method, compcode);

    if (verbosity_level > 1) {
        auto ratio = [](size_t in, size_t out) { return static_cast<double>(in) / static_cast<double>(out); };
        std::cout << "rows: " << rows << ", cols: " << cols << ", nnz: " << nnz << '\n';
        std::cout << "values: " << values_bytes << " bytes in, " << comp_values.size()
                  << " bytes out, ratio = " << ratio(values_bytes, comp_values.size()) << '\n';
        std::cout << "oindex: " << oindex_bytes << " bytes in, " << comp_oindex.size()
                  << " bytes out, ratio = " << ratio(oindex_bytes, comp_oindex.size()) << '\n';
        std::cout << "iindex: " << iindex_bytes << " bytes in, " << comp_iindex.size()
                  << " bytes out, ratio = " << ratio(iindex_bytes, comp_iindex.size()) << std::endl;
    }

    std::ofstream os = open_for_writing(filename);
    write_scalar(os, rows);
    write_scalar(os, cols);
    write_scalar(os, nnz);
    write_scalar(os, static_cast<int>(comp_values.size()));
    write_scalar(os, static_cast<int>(comp_oindex.size()));
    write_scalar(os, static_cast<int>(comp_iindex.size()));
    write_bytes(os, comp_values.data(), comp_values.size());
    write_bytes(os, comp_oindex.data(), comp_oindex.size());
    write_bytes(os, comp_iindex.data(), comp_iindex.size());
}

SparseMatrix io::read_compressed_matrix(std::string const &filename, int verbosity_level) {
    if (verbosity_level > 0) {
        std::cout << "Read compressed matrix " << filename << std::endl;
    }

    std::ifstream is = open_for_reading(filename);

    auto const rows = read_scalar<StorageIndex>(is, "rows");
    auto const cols = read_scalar<StorageIndex>(is, "cols");
    auto const nnz = read_scalar<StorageIndex>(is, "nnz");
    require_non_negative(rows, "number of rows");
    require_non_negative(cols, "number of columns");
    require_non_negative(nnz, "number of non-zeros");

    auto const comp_values_size = read_scalar<int>(is, "compressed values size");
    auto const comp_oindex_size = read_scalar<int>(is, "compressed outer index size");
    auto const comp_iindex_size = read_scalar<int>(is, "compressed inner index size");

    std::vector<char> const comp_values = read_blob(is, comp_values_size, "compressed values");
    std::vector<char> const comp_oindex = read_blob(is, comp_oindex_size, "compressed outer indices");
    std::vector<char> const comp_iindex = read_blob(is, comp_iindex_size, "compressed inner indices");

    // Decompress straight into the matrix's storage.
    SparseMatrix mat(rows, cols);
    mat.makeCompressed();
    mat.resizeNonZeros(nnz);

    decompress(comp_values, mat.valuePtr(), sizeof(float) * nnz);
    decompress(comp_oindex, mat.outerIndexPtr(), sizeof(StorageIndex) * (rows + 1));
    decompress(comp_iindex, mat.innerIndexPtr(), sizeof(StorageIndex) * nnz);

    if (verbosity_level > 1) {
        std::cout << "rows: " << rows << ", cols: " << cols << ", nnz: " << nnz << '\n';
        std::cout << "mat.nonZeros(): " << mat.nonZeros() << ", mat.sum(): " << mat.sum() << std::endl;
    }

    return mat;
}

void io::store_compressed_vector(Eigen::VectorXf const &vec, std::string const &filename,
                                 int compression_level, int shuffle_method, std::string const &codec) {
    check_shuffle_method(shuffle_method);
    int const compcode = compcode_from_name(codec);

    int const rows = static_cast<int>(vec.size());
    std::vector<char> const comp_values = compress(vec.data(), sizeof(float) * rows, sizeof(float),
                                                   compression_level, shuffle_method, compcode);

    std::ofstream os = open_for_writing(filename);
    write_scalar(os, rows);
    write_scalar(os, static_cast<int>(comp_values.size()));
    write_bytes(os, comp_values.data(), comp_values.size());
}

Eigen::VectorXf io::read_compressed_vector(std::string const &filename) {
    std::ifstream is = open_for_reading(filename);

    auto const rows = read_scalar<int>(is, "size");
    if (rows < 0) {
        throw std::runtime_error("Negative vector size stored in file " + filename);
    }
    auto const comp_size = read_scalar<int>(is, "compressed size");
    std::vector<char> const comp_values = read_blob(is, comp_size, "compressed values");

    Eigen::VectorXf vec(rows);
    decompress(comp_values, vec.data(), sizeof(float) * rows);
    return vec;
}

void io::construct_matrix(std::string const &filename, std::vector<int> const &rows,
                          std::vector<int> const &cols, std::vector<float> const &values, int n_rows,
                          int n_cols, int compression_level, int shuffle_method, std::string const &codec,
                          int verbosity_level, int n_threads) {
    if (rows.size() != cols.size() || rows.size() != values.size()) {
        throw std::invalid_argument("rows, cols and values must have the same length");
    }
    omp_set_num_threads(n_threads);

    if (verbosity_level > 0) {
        std::cout << "Construct sparse matrix " << filename << " from " << values.size() << " triplets"
                  << std::endl;
    }

    std::vector<Triplet> triplets;
    triplets.reserve(values.size());

#pragma omp parallel
    {
        std::vector<Triplet> local;
        local.reserve(rows.size() / omp_get_num_threads() + 1);

#pragma omp for schedule(static)
        for (size_t i = 0; i < rows.size(); i++) {
            local.emplace_back(rows[i], cols[i], values[i]);
        }

#pragma omp critical
        triplets.insert(triplets.end(), local.begin(), local.end());
    }

    SparseMatrix mat(n_rows, n_cols);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    mat.makeCompressed();

    if (verbosity_level > 1) {
        std::cout << "mat:\n" << Eigen::MatrixXf(mat) << std::endl;
    }

    store_compressed_matrix(mat, filename, compression_level, shuffle_method, codec, verbosity_level);
}
