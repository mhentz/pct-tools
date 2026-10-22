// Guards against allocations that grow with the number of blocks.
//
// lipschitz::compute_ATAx once allocated one full image per *block* (n_cols x n_blocks
// floats) and reduced them afterwards. With a few thousand blocks and a million voxels that
// is tens of gigabytes. It must only need one image per thread, like gradient_descent().
//
// The check compares the peak resident set size before and after the call. ru_maxrss is a
// process-wide peak, so the bound is only informative if this test's own footprint exceeds
// that of the tests run before it, which the sizes below ensure by a wide margin.

#include "io.h"
#include "lipschitz.h"

#include <filesystem>
#include <string>
#include <sys/resource.h>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace {

constexpr int N_BLOCKS = 256;
constexpr int N_COLS = 4'000'000; // 16 MB per float image
constexpr int N_THREADS = 8;

double peak_rss_bytes() {
    struct rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return static_cast<double>(usage.ru_maxrss);
#else
    return static_cast<double>(usage.ru_maxrss) * 1024.0;
#endif
}

struct Blocks {
    std::vector<std::string> matrices;
    std::filesystem::path dir;

    // One non-zero per block, so the files and the decompressed blocks are tiny and only the
    // image-sized buffers of compute_ATAx contribute to the footprint.
    Blocks() {
        dir = std::filesystem::temp_directory_path() / "pct_tools_memory_test";
        std::filesystem::create_directories(dir);
        for (int i = 0; i < N_BLOCKS; i++) {
            SparseMatrix a(1, N_COLS);
            a.insert(0, i) = 1.0f;
            std::string const m = (dir / (std::to_string(i) + ".mat")).string();
            io::store_compressed_matrix(a, m, 1);
            matrices.push_back(m);
        }
    }
    ~Blocks() { std::filesystem::remove_all(dir); }
};

} // namespace

TEST_CASE("A^T A x needs one image per thread, not per block", "[memory]") {
    Blocks const blocks;
    Eigen::VectorXf const x = Eigen::VectorXf::Ones(N_COLS);

    double const before = peak_rss_bytes();
    Eigen::VectorXf const result = lipschitz::compute_ATAx(x, blocks.matrices, N_THREADS);
    double const growth = peak_rss_bytes() - before;

    REQUIRE(result.size() == N_COLS);
    CHECK(result[0] == 1.0f);
    CHECK(result[N_BLOCKS] == 0.0f);

    // One image per block would be 256 x 16 MB = 4 GB; per thread it is a few hundred MB
    // including Eigen's temporaries.
    double const per_block_bytes = static_cast<double>(N_BLOCKS) * N_COLS * sizeof(float);
    CHECK(growth < per_block_bytes / 4);
}
