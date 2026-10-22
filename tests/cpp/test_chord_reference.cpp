// Chord-length computation checked against an independent reference implementation.
//
// The reference slices every segment at all voxel-boundary planes it crosses and attributes
// each piece to the voxel containing its midpoint. It makes no assumption about how many
// boundaries a segment crosses or in which order, whereas the implementation under test
// handles at most one crossing per axis and rejects segments that cross more. The cases under
// "origin, spacing, step length and ties" pin down the distance to the crossed boundary of
// the grid at the given origin and the handling of two equal distances in the three-crossing
// case.
//
// Voxel indexing follows the implementation: i = floor((y - oy) / st), j = floor((ox - x) / pw),
// k = floor((z - oz) / pw) with origin (ox, oy, oz), pixel width pw and slice thickness st.

#include "exact_chord_matrix.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

namespace {

using Shape = std::tuple<int, int, int>;
using Origin = std::tuple<float, float, float>;
using Dense = std::vector<std::vector<std::vector<float>>>;

Dense zeros(Shape const &shape) {
    auto const [ni, nj, nk] = shape;
    return Dense(ni, std::vector<std::vector<float>>(nj, std::vector<float>(nk, 0.0f)));
}

// Parametric positions t in (0, 1) where a + t (b - a) hits grid_origin + m * spacing.
void plane_crossings(double a, double b, double grid_origin, double spacing, std::vector<double> &ts) {
    if (a == b)
        return;
    double const lo = std::min(a, b), hi = std::max(a, b);
    int const m_lo = static_cast<int>(std::floor((lo - grid_origin) / spacing)) + 1;
    int const m_hi = static_cast<int>(std::ceil((hi - grid_origin) / spacing)) - 1;
    for (int m = m_lo; m <= m_hi; m++) {
        double const t = ((grid_origin + m * spacing) - a) / (b - a);
        if (t > 0.0 && t < 1.0)
            ts.push_back(t);
    }
}

Dense reference_chords(std::vector<float> const &x, std::vector<float> const &y, std::vector<float> const &z,
                       Shape const &shape, Origin const &origin, double pw, double st) {
    auto const [ox, oy, oz] = origin;
    auto const [ni, nj, nk] = shape;
    Dense out = zeros(shape);

    for (size_t s = 0; s + 1 < x.size(); s++) {
        double const p1[3] = {x[s], y[s], z[s]};
        double const p2[3] = {x[s + 1], y[s + 1], z[s + 1]};
        double const d[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
        double const length = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);

        std::vector<double> ts = {0.0, 1.0};
        plane_crossings(p1[0], p2[0], ox, pw, ts);
        plane_crossings(p1[1], p2[1], oy, st, ts);
        plane_crossings(p1[2], p2[2], oz, pw, ts);
        std::sort(ts.begin(), ts.end());
        ts.erase(std::unique(ts.begin(), ts.end()), ts.end());

        for (size_t q = 0; q + 1 < ts.size(); q++) {
            double const tm = 0.5 * (ts[q] + ts[q + 1]);
            double const mid[3] = {p1[0] + tm * d[0], p1[1] + tm * d[1], p1[2] + tm * d[2]};
            int const i = static_cast<int>(std::floor((mid[1] - oy) / st));
            int const j = static_cast<int>(std::floor((ox - mid[0]) / pw));
            int const k = static_cast<int>(std::floor((mid[2] - oz) / pw));
            if (0 <= i && i < ni && 0 <= j && j < nj && 0 <= k && k < nk)
                out[i][j][k] += static_cast<float>((ts[q + 1] - ts[q]) * length);
        }
    }
    return out;
}

Dense from_triplets(std::vector<Triplet> const &triplets, Shape const &shape) {
    auto const [ni, nj, nk] = shape;
    Dense out = zeros(shape);
    for (auto const &t : triplets) {
        REQUIRE(t.row() == 0);
        int const i = t.col() / (nj * nk);
        int const j = (t.col() / nk) % nj;
        int const k = t.col() % nk;
        out[i][j][k] += t.value();
    }
    return out;
}

void check_close(Dense const &actual, Dense const &expected, float atol = 1e-6f) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); i++)
        for (size_t j = 0; j < actual[i].size(); j++)
            for (size_t k = 0; k < actual[i][j].size(); k++) {
                INFO("voxel (" << i << ", " << j << ", " << k << ")");
                CHECK_THAT(actual[i][j][k], Catch::Matchers::WithinAbs(expected[i][j][k], atol));
            }
}

// Largest element-wise difference; one assertion per array instead of one per voxel.
float max_abs_diff(Dense const &a, Dense const &b) {
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); i++)
        for (size_t j = 0; j < a[i].size(); j++)
            for (size_t k = 0; k < a[i][j].size(); k++)
                worst = std::max(worst, std::abs(a[i][j][k] - b[i][j][k]));
    return worst;
}

struct Case {
    std::vector<float> x, y, z;
};

// Image: 2 slices (y), 3 pixels in x (origin_x = 3 -> j = floor(3 - x)), 3 in z.
Shape const SHAPE{2, 3, 3};
Origin const ORIGIN{3.0f, 0.0f, 0.0f};

void check_case(Case const &c, Shape const &shape = SHAPE, Origin const &origin = ORIGIN, float pw = 1.0f,
                float st = 1.0f) {
    Dense const expected = reference_chords(c.x, c.y, c.z, shape, origin, pw, st);

    SECTION("dense") {
        check_close(matrix::compute_matrix_elements(c.x, c.y, c.z, shape, origin, pw, st, false, 1),
                    expected);
    }
    SECTION("triplets") {
        auto const triplets =
            matrix::compute_matrix_triplets({c.x}, {c.y}, {c.z}, shape, origin, pw, st, 0, 1);
        check_close(from_triplets(triplets, shape), expected);
    }
}

} // namespace

// Each case is one two-point segment {x, y, z} in the 2 x 3 x 3 image above, with 1 mm
// voxels, i = floor(y), j = floor(3 - x), k = floor(z). The comments give the planes crossed
// and the order along the segment (t in [0, 1]).
TEST_CASE("chords match the reference for segments with short steps", "[matrix][reference]") {
    SECTION("inside one voxel") {
        // No crossing: the whole length goes to voxel (0, 1, 0).
        check_case({{1.2f, 1.7f}, {0.5f, 0.5f}, {0.2f, 0.7f}});
    }
    SECTION("one boundary in z") {
        // Crosses z = 1 at t = 0.2: the one-boundary branch, split 0.1 / 0.4.
        check_case({{1.5f, 1.5f}, {0.5f, 0.5f}, {0.9f, 1.4f}});
    }
    SECTION("endpoints on boundary") {
        // p1 sits exactly on z = 1. No crossing inside the segment; all of it belongs to k = 1.
        check_case({{1.5f, 1.5f}, {0.5f, 0.5f}, {1.0f, 1.5f}});
    }
    SECTION("zero-length segment") {
        // Three samples, the second equal to the first: the zero-length segment contributes
        // nothing (and must not divide by zero in the chord factors).
        check_case({{1.5f, 1.5f, 1.5f}, {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 1.4f}});
    }
    SECTION("exact 2D corner") {
        // Crosses x = 1 and z = 1 at the same point (t = 0.5): the segment passes through an
        // edge, two equal distances, no intermediate voxel.
        check_case({{1.5f, 0.5f}, {0.5f, 0.5f}, {0.5f, 1.5f}});
    }
    SECTION("exact 3D corner") {
        // Crosses x = 1, y = 1 and z = 1 at t = 0.5: through a corner, three equal distances.
        check_case({{1.5f, 0.5f}, {0.5f, 1.5f}, {0.5f, 1.5f}});
    }
    SECTION("two boundaries xz") {
        // x = 1 at t = 0.5, then z = 1 at t = 0.6: two-boundary branch, intermediate voxel
        // is start shifted in j.
        check_case({{1.1f, 0.9f}, {0.5f, 0.5f}, {0.85f, 1.1f}});
    }
    SECTION("three boundaries xyz") {
        // y = 1 at t = 1/3, z = 1 at t = 0.4, x = 1 at t = 0.5: three-boundary branch, order
        // y, z, x, two intermediate voxels.
        check_case({{1.1f, 0.9f}, {0.95f, 1.1f}, {0.9f, 1.15f}});
    }
    SECTION("exits image") {
        // Crosses z = 3, the far face of the image: the second half lies in k = 3, outside,
        // and is dropped.
        check_case({{1.5f, 1.5f}, {0.5f, 0.5f}, {2.5f, 3.5f}});
    }
    SECTION("negative coordinates outside") {
        // Entirely below the z origin (k = -2 to -1), crossing a plane outside the image:
        // nothing is added and negative indices do not break the bounds checks.
        check_case({{1.5f, 1.5f}, {0.5f, 0.5f}, {-1.5f, -0.5f}});
    }
}

// A curved polyline from (2.5, 0.3, 0.1) to (0.5, 1.7, 2.9) with a sine wiggle in x, sampled
// finely enough that every segment is much shorter than a voxel: many segments of every
// branch above, accumulated in float, hence the looser tolerance.
TEST_CASE("chords match the reference for a smooth 400-step trajectory", "[matrix][reference]") {
    int const n = 400;
    Case c;
    for (int s = 0; s < n; s++) {
        float const t = static_cast<float>(s) / static_cast<float>(n - 1);
        c.x.push_back(2.5f - 2.0f * t + 0.05f * std::sin(7.0f * t));
        c.y.push_back(0.3f + 1.4f * t);
        c.z.push_back(0.1f + 2.8f * t);
    }
    Dense const expected = reference_chords(c.x, c.y, c.z, SHAPE, ORIGIN, 1.0f, 1.0f);
    check_close(matrix::compute_matrix_elements(c.x, c.y, c.z, SHAPE, ORIGIN, 1.0f, 1.0f, false, 1), expected,
                1e-4f);
}

// --- origin, spacing, step length and ties --------------------------------------------------

// Three crossings of which two coincide. The old three-boundary branch only had cases for
// six strict orderings and treated anything else as a corner, giving the middle piece to the
// end voxel.
TEST_CASE("three crossings with two equal distances fill one intermediate voxel", "[matrix][reference]") {
    SECTION("edge first, then the third plane") {
        // x = 1 and y = 1 at t = 0.5 (an edge), then z = 1 at t = 0.75: one intermediate
        // voxel, start shifted in i and j, holding the piece between t = 0.5 and 0.75.
        check_case({{1.5f, 0.5f}, {0.5f, 1.5f}, {0.25f, 1.25f}});
    }
    SECTION("third plane first, then the edge") {
        // z = 1 at t = 0.25, then x = 1 and y = 1 together at t = 0.75: the intermediate voxel
        // is start shifted in k, holding the piece between t = 0.25 and 0.75.
        check_case({{1.75f, 0.75f}, {0.25f, 1.25f}, {0.75f, 1.75f}});
    }
}

// Steps longer than half a voxel but still crossing at most one plane per axis. The old code
// measured each endpoint's distance to its *nearest* plane, which for the far endpoint is the
// plane beyond the one crossed.
TEST_CASE("steps longer than half a voxel with one crossing per axis", "[matrix][reference]") {
    SECTION("long segment past half pixel") {
        // z from 0.9 to 1.7 crosses z = 1; the nearest plane to p2 is z = 2 (0.3 away), not
        // z = 1 (0.7 away).
        check_case({{1.5f, 1.5f}, {0.5f, 0.5f}, {0.9f, 1.7f}});
    }
    SECTION("two boundaries xz, long step") {
        // x = 1 at t = 0.64 and z = 1 at t = 0.67; both components exceed half a voxel.
        check_case({{1.7f, 0.6f}, {0.5f, 0.5f}, {0.4f, 1.3f}});
    }
    SECTION("three boundaries xyz, long step") {
        // z = 1 at t = 0.44, x = 1 at t = 0.57, y = 1 at t = 0.78; the x component is 1.4 mm,
        // the others 0.9 mm.
        check_case({{1.8f, 0.4f}, {0.3f, 1.2f}, {0.6f, 1.5f}});
    }
}

// Image of 1 x 4 x 4 voxels of 0.3 mm with origin (1.2, 0, 0). The segment crosses z = 0.3
// at t = 0.5. The old code reduced the coordinate modulo 1 before reducing modulo the spacing,
// which only works when the spacing divides 1.
TEST_CASE("a spacing that does not divide 1", "[matrix][reference]") {
    check_case({{0.75f, 0.75f}, {0.15f, 0.15f}, {0.1f, 0.5f}}, {1, 4, 4}, {1.2f, 0.0f, 0.0f}, 0.3f, 0.3f);
}

// The old code measured distances from the raw coordinate to planes at integer multiples of
// the spacing, i.e. it assumed the origin was a multiple of the spacing on every axis.
TEST_CASE("the coordinate origin is honoured on every axis", "[matrix][reference]") {
    SECTION("origin along z") {
        // Origin z = 0.1: the crossed plane is z = 1.1 (t = 0.71), not z = 1.
        check_case({{1.5f, 1.5f}, {0.5f, 0.5f}, {0.6f, 1.3f}}, SHAPE, {3.0f, 0.0f, 0.1f});
    }
    SECTION("origin along x and y") {
        // Origin (2.7, -0.4): x planes at 1.7 (crossed at t = 0.5), y planes at 0.6 (crossed
        // at t = 0.67); two-boundary branch, order x then y.
        check_case({{1.8f, 1.6f}, {0.3f, 0.75f}, {0.5f, 0.5f}}, SHAPE, {2.7f, -0.4f, 0.0f});
    }
}

// --- sampling limit ---------------------------------------------------------------------------

// The case analysis handles at most one crossing per axis; coarser steps must fail loudly.
TEST_CASE("segments crossing more than one boundary along an axis are rejected", "[matrix][reference]") {
    // z from 0.3 to 2.6 crosses z = 1 and z = 2 (index change 2).
    Case const two_planes{{1.5f, 1.5f}, {0.5f, 0.5f}, {0.3f, 2.6f}};
    // z from -0.5 to 3.5 crosses all four z planes, entering and leaving the image in one step.
    Case const through_image{{1.5f, 1.5f}, {1.5f, 1.5f}, {-0.5f, 3.5f}};
    for (Case const &c : {two_planes, through_image}) {
        REQUIRE_THROWS_WITH(
            matrix::compute_matrix_elements(c.x, c.y, c.z, SHAPE, ORIGIN, 1.0f, 1.0f, false, 1),
            Catch::Matchers::StartsWith("sampling step too coarse"));
        REQUIRE_THROWS_AS(
            matrix::compute_matrix_triplets({c.x}, {c.y}, {c.z}, SHAPE, ORIGIN, 1.0f, 1.0f, 0, 1),
            std::invalid_argument);
    }
}

// --- sampling independence ------------------------------------------------------------------
//
// Chord lengths are exact for the sampled polyline as long as no segment crosses more than
// one boundary per axis, so a straight trajectory gives the same chords at every step up to
// one voxel, and their sum is the length of the segment clipped to the image. The geometry
// mirrors examples/benchmark_chords.py on a smaller image: "gentle" trajectories run nearly
// along z, "diagonal" ones cross a boundary every millimetre along x and z.

namespace {

Shape const BLOCK_SHAPE{4, 64, 64};
Origin const BLOCK_ORIGIN{64.0f, 0.0f, 0.0f};
float const BLOCK_DEPTH = 64.0f;

struct Line {
    double x0, y0, slope_x, slope_y;
};

Case sample_line(Line const &line, float step) {
    Case c;
    int const n = static_cast<int>(std::lround(BLOCK_DEPTH / step)) + 1;
    for (int s = 0; s < n; s++) {
        double const zz = step * s;
        c.x.push_back(static_cast<float>(line.x0 + line.slope_x * zz));
        c.y.push_back(static_cast<float>(line.y0 + line.slope_y * zz));
        c.z.push_back(static_cast<float>(zz));
    }
    return c;
}

// Length of the straight segment from the first to the last sample that lies inside the
// image box, by clipping the parameter range against the three slabs.
double clipped_length(Case const &c, Shape const &shape, Origin const &origin, double pw, double st) {
    auto const [ox, oy, oz] = origin;
    auto const [ni, nj, nk] = shape;
    double const p1[3] = {c.x.front(), c.y.front(), c.z.front()};
    double const p2[3] = {c.x.back(), c.y.back(), c.z.back()};
    // box limits per axis: x in (ox - nj pw, ox], y in [oy, oy + ni st), z in [oz, oz + nk pw)
    double const lo[3] = {ox - nj * pw, oy, oz};
    double const hi[3] = {ox, oy + ni * st, oz + nk * pw};
    double t_lo = 0.0, t_hi = 1.0;
    for (int a = 0; a < 3; a++) {
        double const d = p2[a] - p1[a];
        if (d == 0.0) {
            if (p1[a] < lo[a] || p1[a] >= hi[a])
                return 0.0;
            continue;
        }
        double t0 = (lo[a] - p1[a]) / d, t1 = (hi[a] - p1[a]) / d;
        if (t0 > t1)
            std::swap(t0, t1);
        t_lo = std::max(t_lo, t0);
        t_hi = std::min(t_hi, t1);
    }
    if (t_hi <= t_lo)
        return 0.0;
    double const dx = p2[0] - p1[0], dy = p2[1] - p1[1], dz = p2[2] - p1[2];
    return (t_hi - t_lo) * std::sqrt(dx * dx + dy * dy + dz * dz);
}

} // namespace

TEST_CASE("chords of straight trajectories do not depend on the sampling step", "[matrix][reference]") {
    std::vector<Line> lines;
    for (int b = 0; b < 12; b++) {
        // gentle: nearly along z
        lines.push_back({4.0 + 56.0 * (b % 5) / 4.0, 0.5 + 3.0 * (b % 4) / 3.0, 0.02 * ((b % 7) - 3), 0.0});
        // diagonal: 45 degrees in x-z, 1:10 in y-z
        lines.push_back({32.0 + 24.0 * ((b % 5) - 2) / 2.0, 0.5 + 3.0 * (b % 4) / 3.0, (b % 2) ? 1.0 : -1.0,
                         0.1 * ((b % 3) - 1)});
    }

    for (size_t l = 0; l < lines.size(); l++) {
        Case const fine = sample_line(lines[l], 0.25f);
        Dense const expected = matrix::compute_matrix_elements(fine.x, fine.y, fine.z, BLOCK_SHAPE,
                                                               BLOCK_ORIGIN, 1.0f, 1.0f, false, 1);
        double const length = clipped_length(fine, BLOCK_SHAPE, BLOCK_ORIGIN, 1.0, 1.0);
        REQUIRE(length > 0.0);

        for (float const step : {0.5f, 1.0f}) {
            INFO("line " << l << ", step " << step);
            Case const coarse = sample_line(lines[l], step);
            auto const triplets = matrix::compute_matrix_triplets(
                {coarse.x}, {coarse.y}, {coarse.z}, BLOCK_SHAPE, BLOCK_ORIGIN, 1.0f, 1.0f, 0, 1);
            Dense const actual = from_triplets(triplets, BLOCK_SHAPE);
            REQUIRE(actual.size() == expected.size());
            CHECK(max_abs_diff(actual, expected) < 1e-4f);

            double total = 0.0;
            for (auto const &t : triplets)
                total += t.value();
            CHECK_THAT(total, Catch::Matchers::WithinRel(length, 1e-5));
        }

        Case const too_coarse = sample_line(lines[l], 4.0f);
        REQUIRE_THROWS_AS(matrix::compute_matrix_triplets({too_coarse.x}, {too_coarse.y}, {too_coarse.z},
                                                          BLOCK_SHAPE, BLOCK_ORIGIN, 1.0f, 1.0f, 0, 1),
                          std::invalid_argument);
    }
}

// ---- TrajectoryBlock view ---------------------------------------------------------------
//
// The Python bindings hand the kernel a view onto contiguous NumPy buffers instead of
// nested vectors (which pybind11 would fill one sample at a time). The view must give the
// same triplets as the vector overload, and reject malformed blocks the same way.

namespace {

bool same_triplets(std::vector<Triplet> const &a, std::vector<Triplet> const &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].row() != b[i].row() || a[i].col() != b[i].col() || a[i].value() != b[i].value()) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST_CASE("a trajectory block viewed through contiguous buffers gives the vector result", "[view]") {
    constexpr int n_traj = 24;
    constexpr int n_points = 65;

    // One contiguous (n_traj x n_points) buffer per axis, as a C-ordered NumPy array would be.
    std::vector<float> xs(n_traj * n_points), ys(n_traj * n_points), zs(n_traj * n_points);
    std::vector<std::vector<float>> x(n_traj), y(n_traj), z(n_traj);
    for (int b = 0; b < n_traj; b++) {
        Line const line{4.0f + 2.5f * static_cast<float>(b), 0.5f + 0.1f * static_cast<float>(b),
                        0.3f * static_cast<float>((b % 5) - 2), 0.02f * static_cast<float>(b % 3)};
        Case const c = sample_line(line, BLOCK_DEPTH / (n_points - 1));
        REQUIRE(static_cast<int>(c.x.size()) == n_points);
        x[b] = c.x;
        y[b] = c.y;
        z[b] = c.z;
        std::copy(c.x.begin(), c.x.end(), xs.begin() + b * n_points);
        std::copy(c.y.begin(), c.y.end(), ys.begin() + b * n_points);
        std::copy(c.z.begin(), c.z.end(), zs.begin() + b * n_points);
    }

    matrix::TrajectoryBlock view;
    view.n_points = n_points;
    for (int b = 0; b < n_traj; b++) {
        view.x.push_back(xs.data() + b * n_points);
        view.y.push_back(ys.data() + b * n_points);
        view.z.push_back(zs.data() + b * n_points);
    }

    auto const from_view = matrix::compute_matrix_triplets(view, BLOCK_SHAPE, BLOCK_ORIGIN, 1.0f, 1.0f, 0, 1);
    auto const from_vectors =
        matrix::compute_matrix_triplets(x, y, z, BLOCK_SHAPE, BLOCK_ORIGIN, 1.0f, 1.0f, 0, 1);
    REQUIRE(!from_view.empty());
    CHECK(same_triplets(from_view, from_vectors));

    SECTION("make_block rejects malformed input") {
        std::vector<std::vector<float>> const short_row = {x[0], std::vector<float>(n_points - 1, 0.0f)};
        std::vector<std::vector<float>> const two_rows = {x[0], x[1]};
        CHECK_THROWS_AS(matrix::make_block(two_rows, two_rows, short_row), std::invalid_argument);
        CHECK_THROWS_AS(matrix::make_block(two_rows, {x[0]}, two_rows), std::invalid_argument);
        CHECK_THROWS_AS(matrix::make_block({}, {}, {}), std::invalid_argument);
        CHECK_THROWS_AS(matrix::make_block({{}}, {{}}, {{}}), std::invalid_argument);
    }
}
