// Multi-threaded results must match the single-threaded ones.
//
// The gradient-descent and dense-chord cases guard against data races in the reductions:
// both functions once accumulated into shared memory from a parallel loop and lost updates.

#include "exact_chord_matrix.h"
#include "gradient.h"
#include "io.h"
#include "lipschitz.h"
#include "residuals.h"

#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace {

constexpr int N_MATRICES = 512;
constexpr int N_COLS = 20000;
constexpr int N_THREADS = 8;
constexpr int N_REPEATS = 10; // a lost update shows up in roughly half the runs, so repeat

struct Pairs {
    std::vector<std::string> matrices;
    std::vector<std::string> vectors;
    std::filesystem::path dir;

    Pairs() {
        dir = std::filesystem::temp_directory_path() / "pct_tools_threading_test";
        std::filesystem::create_directories(dir);
        for (int i = 0; i < N_MATRICES; i++) {
            SparseMatrix a(1, N_COLS);
            a.reserve(N_COLS);
            for (int j = 0; j < N_COLS; j++)
                a.insert(0, j) = 1.0f;
            std::string const m = (dir / (std::to_string(i) + ".mat")).string();
            std::string const v = (dir / ("b_" + std::to_string(i) + ".vec")).string();
            io::store_compressed_matrix(a, m, 1);
            io::store_vector(Eigen::VectorXf::Constant(1, static_cast<float>(i)), v);
            matrices.push_back(m);
            vectors.push_back(v);
        }
    }
    ~Pairs() { std::filesystem::remove_all(dir); }
};

Eigen::VectorXf random_x() {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    Eigen::VectorXf x(N_COLS);
    for (int j = 0; j < N_COLS; j++)
        x[j] = dist(rng);
    return x;
}

bool all_close(Eigen::VectorXf const &a, Eigen::VectorXf const &b, float rtol) {
    return ((a - b).array().abs() <= rtol * b.array().abs()).all();
}

} // namespace

TEST_CASE("gradient descent gives the same result with 1 and 8 threads", "[threading]") {
    Pairs const pairs;
    Eigen::VectorXf const x = random_x();
    Eigen::VectorXf const serial = recon::gradient_descent(x, 1e-3f, pairs.matrices, pairs.vectors, 0, 1);

    bool agree = true;
    for (int r = 0; r < N_REPEATS && agree; r++) {
        Eigen::VectorXf const parallel =
            recon::gradient_descent(x, 1e-3f, pairs.matrices, pairs.vectors, 0, N_THREADS);
        agree = all_close(parallel, serial, 1e-4f);
    }
    CHECK(agree);
}

TEST_CASE("squared residuals give the same result with 1 and 8 threads", "[threading]") {
    Pairs const pairs;
    Eigen::VectorXf const x = random_x();
    float const serial = recon::squared_residuals(x, pairs.matrices, pairs.vectors, 0, 1);
    float const parallel = recon::squared_residuals(x, pairs.matrices, pairs.vectors, 0, N_THREADS);
    CHECK_THAT(parallel, Catch::Matchers::WithinRel(serial, 1e-5f));
}

TEST_CASE("A^T A x gives the same result with 1 and 8 threads", "[threading]") {
    Pairs const pairs;
    Eigen::VectorXf const x = random_x();
    Eigen::VectorXf const serial = lipschitz::compute_ATAx(x, pairs.matrices, 1);
    Eigen::VectorXf const parallel = lipschitz::compute_ATAx(x, pairs.matrices, N_THREADS);
    CHECK(all_close(parallel, serial, 1e-5f));
}

TEST_CASE("dense matrix elements give the same result with 1 and 8 threads", "[threading]") {
    // Many tiny segments inside one voxel: every thread accumulates into the same element.
    int const n_points = 2000000;
    std::vector<float> x(n_points, 0.5f), y(n_points, 0.5f), z(n_points);
    for (int i = 0; i < n_points; i++)
        z[i] = 0.1f + 0.8f * static_cast<float>(i) / static_cast<float>(n_points - 1);

    auto const serial =
        matrix::compute_matrix_elements(x, y, z, {1, 2, 2}, {2.0f, 0.0f, 0.0f}, 1.0f, 1.0f, false, 1);
    auto const parallel =
        matrix::compute_matrix_elements(x, y, z, {1, 2, 2}, {2.0f, 0.0f, 0.0f}, 1.0f, 1.0f, false, N_THREADS);

    REQUIRE_THAT(serial[0][1][0], Catch::Matchers::WithinRel(0.8f, 1e-3f));
    CHECK_THAT(parallel[0][1][0], Catch::Matchers::WithinRel(serial[0][1][0], 1e-3f));
}
