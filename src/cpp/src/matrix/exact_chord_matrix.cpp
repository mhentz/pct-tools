#include "exact_chord_matrix.h"
#include "distance.h"
#include "image.h"
#include "io.h"
#include "parallel.h"
#include "voxel.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Sparse>

using SparseMatrix = Eigen::SparseMatrix<float, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<float>;

// Exact chord lengths of a sampled trajectory through a voxel grid.
//
// A trajectory is a polyline of points (x, y, z). Each segment between consecutive samples
// p1 and p2 is handled independently and its length is split among the voxels it passes
// through. Voxel indices follow
//
//     i = floor((y - origin_y) / slice_thickness)    slices along y
//     j = floor((origin_x - x) / pixel_width)        pixels along x, decreasing with x
//     k = floor((z - origin_z) / pixel_width)        pixels along z
//
// and the column index of the system matrix is the row-major flat index of (i, j, k).
//
// The split is driven by voxel_delta = end - start, the change of voxel index along each
// axis, which must be 0 or +-1: a segment may cross at most one boundary per axis, so
// consecutive samples must be less than one voxel apart along every axis. Segments that
// cross more than one boundary along an axis throw std::invalid_argument (see
// check_sampling_step() below), since the case analysis cannot split them.
//
// For each axis, dxb_1 and dxb_2 are the distances from p1 and p2 to the boundary plane
// crossed along x, which lies at max(j1, j2) * pixel_width from the origin in the
// displacement used for the index (distance_to_crossed_boundary()). chord_x =
// sqrt(1 + (dy/dx)^2 + (dz/dx)^2) converts a distance along x into a distance along the
// segment, so p1_to_xb = dxb_1 * chord_x is the distance from p1 to the x crossing along the
// segment. A zero component gives an infinite chord factor, which is reset to 0 since that
// axis is then never crossed. The cases are
//
//     0 boundaries   the whole segment length goes to the start voxel.
//     1 boundary     p1_to_b goes to the start voxel and p2_to_b to the end voxel.
//     2 boundaries   the crossings are ordered by distance from p1. The start voxel gets the
//                    distance to the first crossing, the end voxel the distance from p2 to the
//                    second, and the voxel in between (start shifted along the first crossed
//                    axis) gets the difference. Equal distances mean the segment passes
//                    exactly through an edge and no intermediate voxel is filled.
//     3 boundaries   as above with two intermediate voxels. Two equal distances (the segment
//                    passes through an edge and crosses the third plane before or after it)
//                    fill one intermediate voxel; three equal distances (a corner) fill none.
//
// Segments with both endpoints outside the grid and at most one crossing are skipped; all
// other contributions to voxels outside the grid are dropped at insertion.
//
// compute_matrix_elements() accumulates the chords of a single trajectory into a dense image
// (one per thread, summed after the loop so that no two threads add to the same voxel) and is
// meant for testing and visualisation. compute_matrix_triplets() applies the same logic to a
// block of trajectories and emits sparse (row, column, value) triplets instead. The two
// bodies are deliberately kept as separate, near-identical copies so that the dense version
// can print every intermediate quantity without slowing down the production path.

namespace {

/// The case analysis handles at most one crossed boundary per axis. Throws
/// std::invalid_argument when two consecutive samples are further apart than that, i.e. the
/// step is too coarse. `trajectory` is the row of the trajectory in the block, or -1 for a
/// single trajectory.
void check_sampling_step(Voxel const &voxel_delta, int const segment, int const trajectory) {
    if (std::abs(voxel_delta.i()) <= 1 && std::abs(voxel_delta.j()) <= 1 && std::abs(voxel_delta.k()) <= 1) {
        return;
    }
    std::ostringstream message;
    message << "sampling step too coarse: segment " << segment;
    if (trajectory >= 0) {
        message << " of trajectory " << trajectory;
    }
    message << " changes the voxel index by " << voxel_delta
            << "; consecutive samples must be less than one voxel apart along every axis";
    throw std::invalid_argument(message.str());
}

} // namespace

std::vector<std::vector<std::vector<float>>>
matrix::compute_matrix_elements(std::vector<float> const &x, std::vector<float> const &y,
                                std::vector<float> const &z, std::tuple<int, int, int> const img_shape,
                                std::tuple<float, float, float> const coord_origin, float const pixel_width,
                                float const slice_thickness, bool is_verbose, int const n_threads) {
    omp_set_num_threads(n_threads);

    auto const n_z = static_cast<int>(z.size());

    // Displacements from the origin are formed in double so that the float coordinates are not
    // rounded a second time before the boundary distances are taken.
    double const origin_x = std::get<0>(coord_origin);
    double const origin_y = std::get<1>(coord_origin);
    double const origin_z = std::get<2>(coord_origin);

    const int i_max = std::get<0>(img_shape);
    const int j_max = std::get<1>(img_shape);
    const int k_max = std::get<2>(img_shape);

    // Used for bounds checks and flat indices only; the chords are accumulated below.
    Image const img(0, 0, 0, i_max, j_max, k_max);
    auto const n_voxels =
        static_cast<size_t>(i_max) * static_cast<size_t>(j_max) * static_cast<size_t>(k_max);
    std::vector<float> chords(n_voxels, 0.0F);

    // Per-segment quantities are precomputed in simple vectorisable loops; the case analysis
    // below then only reads them.
    std::vector<float> dx_arr(n_z - 1);
    std::vector<float> dy_arr(n_z - 1);
    std::vector<float> dz_arr(n_z - 1);
#pragma omp parallel for simd
    for (int i = 0; i < n_z - 1; i++) {
        dx_arr[i] = x[i + 1] - x[i];
        dy_arr[i] = y[i + 1] - y[i];
        dz_arr[i] = z[i + 1] - z[i];
    }

    std::vector<int> i1_arr(n_z - 1);
    std::vector<int> j1_arr(n_z - 1);
    std::vector<int> k1_arr(n_z - 1);
#pragma omp parallel for
    for (int i = 0; i < n_z - 1; i++) {
        i1_arr[i] = voxel_index(y[i] - origin_y, slice_thickness);
        j1_arr[i] = voxel_index(origin_x - x[i], pixel_width);
        k1_arr[i] = voxel_index(z[i] - origin_z, pixel_width);
    }

    std::vector<int> i2_arr(n_z - 1);
    std::vector<int> j2_arr(n_z - 1);
    std::vector<int> k2_arr(n_z - 1);
#pragma omp parallel for
    for (int i = 0; i < n_z - 1; i++) {
        i2_arr[i] = voxel_index(y[i + 1] - origin_y, slice_thickness);
        j2_arr[i] = voxel_index(origin_x - x[i + 1], pixel_width);
        k2_arr[i] = voxel_index(z[i + 1] - origin_z, pixel_width);
    }

    // Distances from each endpoint to the boundary plane crossed along each axis; only
    // meaningful, and only used, where the voxel index changes along that axis.
    std::vector<float> dyb1_arr(n_z - 1);
    std::vector<float> dxb1_arr(n_z - 1);
    std::vector<float> dzb1_arr(n_z - 1);
#pragma omp parallel for
    for (int i = 0; i < n_z - 1; i++) {
        dyb1_arr[i] = distance_to_crossed_boundary(y[i] - origin_y, i1_arr[i], i2_arr[i], slice_thickness);
        dxb1_arr[i] = distance_to_crossed_boundary(origin_x - x[i], j1_arr[i], j2_arr[i], pixel_width);
        dzb1_arr[i] = distance_to_crossed_boundary(z[i] - origin_z, k1_arr[i], k2_arr[i], pixel_width);
    }

    std::vector<float> dyb2_arr(n_z - 1);
    std::vector<float> dxb2_arr(n_z - 1);
    std::vector<float> dzb2_arr(n_z - 1);
#pragma omp parallel for
    for (int i = 0; i < n_z - 1; i++) {
        dyb2_arr[i] =
            distance_to_crossed_boundary(y[i + 1] - origin_y, i1_arr[i], i2_arr[i], slice_thickness);
        dxb2_arr[i] = distance_to_crossed_boundary(origin_x - x[i + 1], j1_arr[i], j2_arr[i], pixel_width);
        dzb2_arr[i] = distance_to_crossed_boundary(z[i + 1] - origin_z, k1_arr[i], k2_arr[i], pixel_width);
    }

    bool const debug_condition = is_verbose;

    ExceptionCollector errors;

    // Each thread accumulates into its own image and the images are summed once the loop is
    // over, so two threads never add to the same voxel.
#pragma omp parallel
    {
        Image local(0, 0, 0, i_max, j_max, k_max);
        local.initialise_matrix();

#pragma omp for
        for (int i = 0; i < n_z - 1; i++) {
            errors.run([&] {
                if (debug_condition) {
                    std::cout << '\n';
                    std::cout << '\n';
                    std::cout << "----> DEBUG STEP " << i << " <----" << '\n';
                    std::cout << '\n';
                }

                float const dx = dx_arr[i];
                float const dy = dy_arr[i];
                float const dz = dz_arr[i];

                Voxel const start = Voxel(i1_arr[i], j1_arr[i], k1_arr[i]);
                Voxel const end = Voxel(i2_arr[i], j2_arr[i], k2_arr[i]);

                Voxel mid = Voxel(i1_arr[i], j1_arr[i], k1_arr[i]);
                Voxel voxel_delta = end - start;

                check_sampling_step(voxel_delta, i, -1);

                float const dyb_1 = dyb1_arr[i];
                float const dxb_1 = dxb1_arr[i];
                float const dzb_1 = dzb1_arr[i];

                float const dyb_2 = dyb2_arr[i];
                float const dxb_2 = dxb2_arr[i];
                float const dzb_2 = dzb2_arr[i];

                if (debug_condition) {
                    std::cout << "position" << '\n';
                    std::cout << "------------------------" << '\n';
                    std::cout << "r1: (" << x[i] << ", " << y[i] << ", " << z[i] << ")" << '\n';
                    std::cout << "r2: (" << x[i + 1] << ", " << y[i + 1] << ", " << z[i + 1] << ")" << '\n';
                    std::cout << "dr: (" << dx << ", " << dy << ", " << dz << ")" << '\n';
                    std::cout << '\n';

                    std::cout << "voxel values" << '\n';
                    std::cout << "------------------------" << '\n';
                    std::cout << "start: " << start << '\n';
                    std::cout << "end: " << end << '\n';
                    std::cout << "delta: " << voxel_delta << '\n';
                    std::cout << "mid: " << mid << '\n';
                    std::cout << '\n';

                    std::cout << "distances to boundaries" << '\n';
                    std::cout << "------------------------" << '\n';
                    std::cout << "dist_bound1: (" << dxb_1 << ", " << dyb_1 << ", " << dzb_1 << ")" << '\n';
                    std::cout << "dist_bound2: (" << dxb_2 << ", " << dyb_2 << ", " << dzb_2 << ")" << '\n';
                    std::cout << '\n';
                }

                if (!start.is_in(img) && !end.is_in(img) &&
                    (abs(voxel_delta.i()) + abs(voxel_delta.j()) + abs(voxel_delta.k()) < 2)) {
                    return;
                }

                // crossing no boundaries
                if (voxel_delta.i() == 0 && voxel_delta.j() == 0 && voxel_delta.k() == 0) {
                    local.add_value(start, distance(dx, dy, dz));

                    if (debug_condition) {
                        std::cout << "NO BOUDARIES CROSSED" << '\n';
                        std::cout << "chord: " << distance(dx, dy, dz) << '\n';
                        std::cout << '\n';
                    }
                    return;
                }

                float chord_in_pixel1 = NAN;
                float chord_in_pixel2 = NAN;

                float chord_x = std::sqrt(1 + ((dy / dx) * (dy / dx)) + ((dz / dx) * (dz / dx)));
                float chord_y = std::sqrt(1 + ((dx / dy) * (dx / dy)) + ((dz / dy) * (dz / dy)));
                float chord_z = std::sqrt(1 + ((dx / dz) * (dx / dz)) + ((dy / dz) * (dy / dz)));

                if (!std::isfinite(chord_x)) {
                    chord_x = 0.0;
                }

                if (!std::isfinite(chord_y)) {
                    chord_y = 0.0;
                }

                if (!std::isfinite(chord_z)) {
                    chord_z = 0.0;
                }

                if (debug_condition) {
                    std::cout << "CHORDS" << '\n';
                    std::cout << "------------------------" << '\n';
                    std::cout << "c: (" << chord_x << ", " << chord_y << ", " << chord_z << ")" << '\n';
                    std::cout << '\n';
                }

                int const n_boundaries_crossed =
                    std::abs(voxel_delta.i()) + std::abs(voxel_delta.j()) + std::abs(voxel_delta.k());

                // crossed 1 boundary
                if (n_boundaries_crossed == 1) {

                    if (voxel_delta.i() != 0) {
                        chord_in_pixel1 = dyb_1 * chord_y;
                        chord_in_pixel2 = dyb_2 * chord_y;
                    } else if (voxel_delta.j() != 0) {
                        chord_in_pixel1 = dxb_1 * chord_x;
                        chord_in_pixel2 = dxb_2 * chord_x;
                    } else {
                        chord_in_pixel1 = dzb_1 * chord_z;
                        chord_in_pixel2 = dzb_2 * chord_z;
                    }

                    local.add_value(start, chord_in_pixel1);
                    local.add_value(end, chord_in_pixel2);

                    if (debug_condition) {
                        std::cout << "CROSSED ONE BOUNDARY" << '\n';
                        std::cout << "------------------------" << '\n';
                        std::cout << "chord_in_pixel1: " << chord_in_pixel1 << '\n';
                        std::cout << "chord_in_pixel2: " << chord_in_pixel2 << '\n';
                    }
                    return;
                }

                // crossing more than 1 boundary
                float const p1_to_xb = dxb_1 * chord_x;
                float const p1_to_yb = dyb_1 * chord_y;
                float const p1_to_zb = dzb_1 * chord_z;

                float const p2_to_xb = dxb_2 * chord_x;
                float const p2_to_yb = dyb_2 * chord_y;
                float const p2_to_zb = dzb_2 * chord_z;

                if (debug_condition) {
                    std::cout << "p1_to_xb: " << p1_to_xb << '\n';
                    std::cout << "p1_to_yb: " << p1_to_yb << '\n';
                    std::cout << "p1_to_zb: " << p1_to_zb << '\n';

                    std::cout << "p2_to_xb: " << p2_to_xb << '\n';
                    std::cout << "p2_to_yb: " << p2_to_yb << '\n';
                    std::cout << "p2_to_zb: " << p2_to_zb << '\n';
                }

                if (voxel_delta.i() != 0 && voxel_delta.j() == 0 && voxel_delta.k() != 0) {

                    std::string condition;
                    bool fill_mid = true;

                    if (p1_to_yb < p1_to_zb) {
                        condition = "CONDITION 1: p1_to_yb < p1_to_zb";
                        chord_in_pixel1 = p1_to_yb;
                        chord_in_pixel2 = p2_to_zb;

                        mid += Voxel(voxel_delta.i(), 0, 0);
                    } else if (p1_to_zb < p1_to_yb) {
                        condition = "CONDITION 2: p1_to_zb < p1_to_yb";
                        chord_in_pixel1 = p1_to_zb;
                        chord_in_pixel2 = p2_to_yb;

                        mid += Voxel(0, 0, voxel_delta.k());
                    } else {
                        condition = "CONDITION 3: p1_to_yb = p1_to_zb";
                        chord_in_pixel1 = p1_to_zb;
                        chord_in_pixel2 = p2_to_zb;

                        fill_mid = false;
                    }

                    if (fill_mid) {
                        float const yb_to_zb = std::abs(p1_to_yb - p1_to_zb);
                        local.add_value(mid, yb_to_zb);
                    }

                    if (debug_condition) {
                        float const yb_to_zb = std::abs(p1_to_yb - p1_to_zb);

                        std::cout << '\n';
                        std::cout << condition << '\n';
                        std::cout << '\n';

                        std::cout << "CROSSED I AND K BOUNDARY" << '\n';
                        std::cout << "------------------------" << '\n';
                        std::cout << "distance from p1 to boundaries" << '\n';
                        std::cout << "  p1_to_yb: " << p1_to_yb << '\n';
                        std::cout << "  p1_to_zb: " << p1_to_zb << '\n';

                        std::cout << "distance from p2 to boundaries" << '\n';
                        std::cout << "  p2_to_yb: " << p2_to_yb << '\n';
                        std::cout << "  p2_to_zb: " << p2_to_zb << '\n';

                        std::cout << "distance between boundaries" << '\n';
                        std::cout << "  yb_to_zb: " << yb_to_zb << '\n';

                        std::cout << "distance in each pixel" << '\n';
                        std::cout << "  chord_in_pixel1: " << chord_in_pixel1 << '\n';
                        std::cout << "  chord_in_pixel2: " << chord_in_pixel2 << '\n';
                        std::cout << "  yb_to_zb: " << yb_to_zb << '\n';
                        std::cout << "  sum: " << chord_in_pixel1 + chord_in_pixel2 + yb_to_zb << '\n';

                        std::cout << "difference between sum and expected" << '\n';
                        std::cout << "  diff: "
                                  << (chord_in_pixel1 + chord_in_pixel2 + yb_to_zb) -
                                         std::sqrt(std::pow(dx, 2) + std::pow(dz, 2) + std::pow(dy, 2))
                                  << '\n';

                        std::cout << "inter pixel: " << mid << '\n';
                        std::cout << '\n';
                    }
                }

                else if (voxel_delta.i() == 0 && voxel_delta.j() != 0 && voxel_delta.k() != 0) {

                    std::string condition;
                    bool fill_mid = true;

                    if (p1_to_xb < p1_to_zb) {
                        condition = "CONDITION 1: p1_to_xb < p1_to_zb";

                        chord_in_pixel1 = p1_to_xb;
                        chord_in_pixel2 = p2_to_zb;

                        mid += Voxel(0, voxel_delta.j(), 0);
                    } else if (p1_to_zb < p1_to_xb) {
                        condition = "CONDITION 2: p1_to_zb < p1_to_xb";

                        chord_in_pixel1 = p1_to_zb;
                        chord_in_pixel2 = p2_to_xb;

                        mid += Voxel(0, 0, voxel_delta.k());
                    } else {
                        condition = "CONDITION 3: p1_to_xb = p1_to_zb";

                        chord_in_pixel1 = p1_to_xb;
                        chord_in_pixel2 = p2_to_xb;

                        fill_mid = false;
                    }

                    if (fill_mid) {
                        // distance between the two boundary intersections
                        float xb_to_zb = std::abs(p1_to_xb - p1_to_zb);
                        local.add_value(mid, xb_to_zb);
                    }

                    if (debug_condition) {
                        std::cout << '\n';
                        std::cout << condition << '\n';
                        std::cout << '\n';

                        float xb_to_zb = std::abs(p1_to_xb - p1_to_zb);

                        std::cout << "CROSSED J AND K BOUNDARY" << '\n';
                        std::cout << "------------------------" << '\n';
                        std::cout << "distance from p1 to boundaries" << '\n';
                        std::cout << "  p1_to_xb: " << p1_to_xb << '\n';
                        std::cout << "  p1_to_zb: " << p1_to_zb << '\n';

                        std::cout << "distance from p2 to boundaries" << '\n';
                        std::cout << "  p2_to_xb: " << p2_to_xb << '\n';
                        std::cout << "  p2_to_zb: " << p2_to_zb << '\n';

                        std::cout << "distance between boundaries" << '\n';
                        std::cout << "  xb_to_zb: " << xb_to_zb << '\n';

                        std::cout << "distance in each pixel" << '\n';
                        std::cout << "  chord_in_pixel1: " << chord_in_pixel1 << '\n';
                        std::cout << "  chord_in_pixel2: " << chord_in_pixel2 << '\n';
                        std::cout << "  xb_to_zb: " << xb_to_zb << '\n';
                        std::cout << "  sum: " << chord_in_pixel1 + chord_in_pixel2 + xb_to_zb << '\n';

                        std::cout << "difference between sum and expected" << '\n';
                        std::cout << "  diff: "
                                  << (chord_in_pixel1 + chord_in_pixel2 + xb_to_zb) -
                                         std::sqrt(std::pow(dx, 2) + std::pow(dz, 2) + std::pow(dy, 2))
                                  << '\n';

                        std::cout << "inter pixel: " << mid << '\n';
                        std::cout << '\n';
                    }

                }

                else if (voxel_delta.i() != 0 && voxel_delta.j() != 0 && voxel_delta.k() == 0) {

                    std::string condition;
                    bool fill_mid = true;

                    if (p1_to_yb < p1_to_xb) {
                        condition = "CONDITION 1: p1_to_yb < p1_to_xb";
                        chord_in_pixel1 = p1_to_yb;
                        chord_in_pixel2 = p2_to_xb;

                        mid += Voxel(voxel_delta.i(), 0, 0);
                    } else if (p1_to_xb < p1_to_yb) {
                        condition = "CONDITION 2: p1_to_xb < p1_to_yb";
                        chord_in_pixel1 = p1_to_xb;
                        chord_in_pixel2 = p2_to_yb;

                        mid += Voxel(0, voxel_delta.j(), 0);
                    } else {
                        condition = "CONDITION 3: p1_to_yb = p1_to_xb";
                        chord_in_pixel1 = p1_to_yb;
                        chord_in_pixel2 = p2_to_yb;

                        fill_mid = false;
                    }

                    if (fill_mid) {
                        float const yb_to_xb = std::abs(p1_to_yb - p1_to_xb);
                        local.add_value(mid, yb_to_xb);
                    }

                    if (debug_condition) {
                        std::cout << '\n';
                        std::cout << condition << '\n';
                        std::cout << '\n';

                        float const yb_to_xb = std::abs(p1_to_yb - p1_to_xb);

                        std::cout << "CROSSED I AND J BOUNDARY" << '\n';
                        std::cout << "------------------------" << '\n';
                        std::cout << "distance from p1 to boundaries" << '\n';
                        std::cout << "  p1_to_xb: " << p1_to_xb << '\n';
                        std::cout << "  p1_to_yb: " << p1_to_yb << '\n';

                        std::cout << "distance from p2 to boundaries" << '\n';
                        std::cout << "  p2_to_xb: " << p2_to_xb << '\n';
                        std::cout << "  p2_to_yb: " << p2_to_yb << '\n';

                        std::cout << "distance between boundaries" << '\n';
                        std::cout << "  yb_to_xb: " << yb_to_xb << '\n';

                        std::cout << "distance in each pixel" << '\n';
                        std::cout << "  chord_in_pixel1: " << chord_in_pixel1 << '\n';
                        std::cout << "  chord_in_pixel2: " << chord_in_pixel2 << '\n';
                        std::cout << "  yb_to_xb: " << yb_to_xb << '\n';
                        std::cout << "  sum: " << chord_in_pixel1 + chord_in_pixel2 + yb_to_xb << '\n';

                        std::cout << "difference between sum and expected" << '\n';
                        std::cout << "  diff: "
                                  << (chord_in_pixel1 + chord_in_pixel2 + yb_to_xb) -
                                         std::sqrt(std::pow(dx, 2) + std::pow(dz, 2) + std::pow(dy, 2))
                                  << '\n';

                        std::cout << "inter pixel: " << mid << '\n';
                        std::cout << '\n';
                    }
                }

                // crossing 3 boundaries
                else {
                    float const xb_to_yb = std::abs(p1_to_xb - p1_to_yb);
                    float const yb_to_zb = std::abs(p1_to_yb - p1_to_zb);
                    float const xb_to_zb = std::abs(p1_to_xb - p1_to_zb);

                    Voxel mid1 = start;
                    Voxel mid2 = start;

                    float mid1_value = 0.0;
                    float mid2_value = 0.0;

                    std::string condition;
                    bool fill_mid = true;

                    if ((p1_to_xb < p1_to_yb) & (p1_to_yb < p1_to_zb)) {
                        condition = "CONDITION 1: p1_to_xb < p1_to_yb < p1_to_zb";

                        chord_in_pixel1 = p1_to_xb;
                        chord_in_pixel2 = p2_to_zb;

                        mid1 += Voxel(0, voxel_delta.j(), 0);
                        mid2 += Voxel(voxel_delta.i(), voxel_delta.j(), 0);

                        mid1_value = xb_to_yb;
                        mid2_value = yb_to_zb;

                    } else if ((p1_to_xb < p1_to_zb) & (p1_to_zb < p1_to_yb)) {
                        condition = "CONDITION 2: p1_to_xb < p1_to_zb < p1_to_yb";

                        chord_in_pixel1 = p1_to_xb;
                        chord_in_pixel2 = p2_to_yb;

                        mid1 += Voxel(0, voxel_delta.j(), 0);
                        mid2 += Voxel(0, voxel_delta.j(), voxel_delta.k());

                        mid1_value = xb_to_zb;
                        mid2_value = yb_to_zb;

                    } else if ((p1_to_yb < p1_to_xb) & (p1_to_xb < p1_to_zb)) {
                        condition = "CONDITION 3: p1_to_yb < p1_to_xb < p1_to_zb";

                        chord_in_pixel1 = p1_to_yb;
                        chord_in_pixel2 = p2_to_zb;

                        mid1 += Voxel(voxel_delta.i(), 0, 0);
                        mid2 += Voxel(voxel_delta.i(), voxel_delta.j(), 0);

                        mid1_value = xb_to_yb;
                        mid2_value = xb_to_zb;

                    } else if ((p1_to_yb < p1_to_zb) & (p1_to_zb < p1_to_xb)) {
                        condition = "CONDITION 4: p1_to_yb < p1_to_zb < p1_to_xb";

                        chord_in_pixel1 = p1_to_yb;
                        chord_in_pixel2 = p2_to_xb;

                        mid1 += Voxel(voxel_delta.i(), 0, 0);
                        mid2 += Voxel(voxel_delta.i(), 0, voxel_delta.k());

                        mid1_value = yb_to_zb;
                        mid2_value = xb_to_zb;

                    } else if ((p1_to_zb < p1_to_xb) & (p1_to_xb < p1_to_yb)) {
                        condition = "CONDITION 5: p1_to_zb < p1_to_xb < p1_to_yb";

                        chord_in_pixel1 = p1_to_zb;
                        chord_in_pixel2 = p2_to_yb;

                        mid1 += Voxel(0, 0, voxel_delta.k());
                        mid2 += Voxel(0, voxel_delta.j(), voxel_delta.k());

                        mid1_value = xb_to_zb;
                        mid2_value = xb_to_yb;

                    } else if ((p1_to_zb < p1_to_yb) & (p1_to_yb < p1_to_xb)) {
                        condition = "CONDITION 6: p1_to_zb < p1_to_yb < p1_to_xb";

                        chord_in_pixel1 = p1_to_zb;
                        chord_in_pixel2 = p2_to_xb;

                        mid1 += Voxel(0, 0, voxel_delta.k());
                        mid2 += Voxel(voxel_delta.i(), 0, voxel_delta.k());

                        mid1_value = yb_to_zb;
                        mid2_value = xb_to_yb;

                    } else {
                        // Two of the distances are equal (the segment passes through an edge and
                        // crosses the third plane before or after it) or all three are (a corner).
                        // At most one intermediate voxel is filled: start shifted along every axis
                        // whose crossing comes first. mid2 stays at start with a zero chord.
                        float const p1_to_first = std::min({p1_to_xb, p1_to_yb, p1_to_zb});
                        float const p1_to_last = std::max({p1_to_xb, p1_to_yb, p1_to_zb});

                        chord_in_pixel1 = p1_to_first;
                        chord_in_pixel2 = (p1_to_last == p1_to_xb)   ? p2_to_xb
                                          : (p1_to_last == p1_to_yb) ? p2_to_yb
                                                                     : p2_to_zb;

                        if (p1_to_first < p1_to_last) {
                            condition = "CONDITION 7: two equal distances (edge)";

                            mid1 += Voxel(p1_to_yb == p1_to_first ? voxel_delta.i() : 0,
                                          p1_to_xb == p1_to_first ? voxel_delta.j() : 0,
                                          p1_to_zb == p1_to_first ? voxel_delta.k() : 0);
                            mid1_value = p1_to_last - p1_to_first;
                        } else {
                            condition = "CONDITION 8: three equal distances (corner)";

                            fill_mid = false;
                        }
                    }

                    if (fill_mid) {
                        local.add_value(mid1, mid1_value);
                        local.add_value(mid2, mid2_value);
                    }

                    if (debug_condition) {
                        std::cout << "mid1: " << mid1 << '\n';
                        std::cout << "mid2: " << mid2 << '\n';
                        std::cout << "mid1_values: " << mid1_value << '\n';
                        std::cout << "mid2_values: " << mid2_value << '\n';

                        std::cout << '\n';
                        std::cout << condition << '\n';
                        std::cout << '\n';

                        std::cout << "CROSSED I, J, AND K BOUNDARY" << '\n';
                        std::cout << "------------------------" << '\n';
                        std::cout << "distance from p1 to boundaries" << '\n';
                        std::cout << "  p1_to_xb: " << p1_to_xb << '\n';
                        std::cout << "  p1_to_yb: " << p1_to_yb << '\n';
                        std::cout << "  p1_to_zb: " << p1_to_zb << '\n';

                        std::cout << "distance from p2 to boundaries" << '\n';
                        std::cout << "  p2_to_xb: " << p2_to_xb << '\n';
                        std::cout << "  p2_to_yb: " << p2_to_yb << '\n';
                        std::cout << "  p2_to_zb: " << p2_to_zb << '\n';

                        std::cout << "distance between boundaries" << '\n';
                        std::cout << "  xb_to_yb: " << xb_to_yb << '\n';
                        std::cout << "  yb_to_zb: " << yb_to_zb << '\n';

                        std::cout << "distance in each pixel" << '\n';
                        std::cout << "  chord_in_pixel1: " << chord_in_pixel1 << '\n';
                        std::cout << "  chord_in_pixel2: " << chord_in_pixel2 << '\n';
                        std::cout << "  xb_to_yb: " << xb_to_yb << '\n';
                        std::cout << "  yb_to_zb: " << yb_to_zb << '\n';
                        std::cout << "  sum: " << chord_in_pixel1 + chord_in_pixel2 + xb_to_yb + yb_to_zb
                                  << '\n';

                        std::cout << "difference between sum and expected" << '\n';
                        std::cout << "  diff: "
                                  << (chord_in_pixel1 + chord_in_pixel2 + xb_to_yb + yb_to_zb) -
                                         std::sqrt(std::pow(dx, 2) + std::pow(dz, 2) + std::pow(dy, 2))
                                  << '\n';

                        std::cout << "inter pixel: " << mid << '\n';
                        std::cout << '\n';
                    }
                }

                local.add_value(start, chord_in_pixel1);
                local.add_value(end, chord_in_pixel2);
            });
        }

#pragma omp critical(pct_tools_dense_chords_reduce)
        for (size_t idx = 0; idx < n_voxels; idx++) {
            chords[idx] += local.mat()[idx];
        }
    }
    errors.rethrow();

    // construct 3d vector
    std::vector<std::vector<std::vector<float>>> img_vec;

    for (int i = 0; i < i_max; i++) {
        std::vector<std::vector<float>> img_vec_2;

        for (int j = 0; j < j_max; j++) {
            std::vector<float> img_vec_3;

            for (int k = 0; k < k_max; k++) {
                long long index = (i * j_max * k_max) + (j * k_max) + k;
                img_vec_3.push_back(chords[index]);
            }

            img_vec_2.push_back(img_vec_3);
        }
        img_vec.push_back(img_vec_2);
    }

    return img_vec;
}

matrix::TrajectoryBlock matrix::make_block(std::vector<std::vector<float>> const &x_array,
                                           std::vector<std::vector<float>> const &y_array,
                                           std::vector<std::vector<float>> const &z_array) {
    if ((x_array.size() != y_array.size()) || (x_array.size() != z_array.size())) {
        throw std::invalid_argument("input arrays must have the same number of trajectories");
    }
    if (x_array.empty()) {
        throw std::invalid_argument("input arrays must contain at least one trajectory");
    }

    auto const block_size = static_cast<int>(x_array.size());
    auto const n_points = static_cast<int>(x_array[0].size());

    if (n_points == 0) {
        throw std::invalid_argument("input x_array must have 2 dimensions");
    }

    TrajectoryBlock block;
    block.n_points = n_points;
    block.x.reserve(block_size);
    block.y.reserve(block_size);
    block.z.reserve(block_size);
    for (int b = 0; b < block_size; b++) {
        if ((static_cast<int>(x_array[b].size()) != n_points) ||
            (static_cast<int>(y_array[b].size()) != n_points) ||
            (static_cast<int>(z_array[b].size()) != n_points)) {
            throw std::invalid_argument("input arrays must have the same size");
        }
        block.x.push_back(x_array[b].data());
        block.y.push_back(y_array[b].data());
        block.z.push_back(z_array[b].data());
    }
    return block;
}

std::vector<Triplet> matrix::compute_matrix_triplets(std::vector<std::vector<float>> const &x_array,
                                                     std::vector<std::vector<float>> const &y_array,
                                                     std::vector<std::vector<float>> const &z_array,
                                                     std::tuple<int, int, int> const img_shape,
                                                     std::tuple<float, float, float> const coord_origin,
                                                     float const pixel_width, float const slice_thickness,
                                                     int const verbosity_level, int const n_threads) {
    return compute_matrix_triplets(make_block(x_array, y_array, z_array), img_shape, coord_origin,
                                   pixel_width, slice_thickness, verbosity_level, n_threads);
}

std::vector<Triplet> matrix::compute_matrix_triplets(TrajectoryBlock const &block,
                                                     std::tuple<int, int, int> const img_shape,
                                                     std::tuple<float, float, float> const coord_origin,
                                                     float const pixel_width, float const slice_thickness,
                                                     int verbosity_level, int const n_threads) {
    if ((block.x.size() != block.y.size()) || (block.x.size() != block.z.size())) {
        throw std::invalid_argument("input arrays must have the same number of trajectories");
    }
    if (block.x.empty()) {
        throw std::invalid_argument("input arrays must contain at least one trajectory");
    }
    if (block.n_points <= 0) {
        throw std::invalid_argument("input x_array must have 2 dimensions");
    }

    omp_set_num_threads(n_threads);

    int const block_size = block.size();
    int const sub_array_size = block.n_points;
    int const n_total_elements = sub_array_size * block_size;

    if (verbosity_level > 0) {
        std::cout << "block_size: " << block_size << '\n';
        std::cout << "sub_array_size: " << sub_array_size << '\n';
        std::cout << "n_total_elements: " << n_total_elements << '\n';
        std::cout << '\n';
    }

    // Displacements from the origin are formed in double so that the float coordinates are not
    // rounded a second time before the boundary distances are taken.
    double const origin_x = std::get<0>(coord_origin);
    double const origin_y = std::get<1>(coord_origin);
    double const origin_z = std::get<2>(coord_origin);

    int const i_max = std::get<0>(img_shape);
    int const j_max = std::get<1>(img_shape);
    int const k_max = std::get<2>(img_shape);

    if (verbosity_level > 1) {
        std::cout << "origin: (" << origin_x << ", " << origin_y << ", " << origin_z << ")" << '\n';
        std::cout << "img_shape: (" << i_max << ", " << j_max << ", " << k_max << ")" << '\n';
        std::cout << '\n';
    }

    // Reserve for roughly three voxels per pixel column of the image per trajectory. This is
    // only a capacity hint; the vector grows if a block needs more.
    int const n_nonzero_approx = j_max * block_size * 3;

    if (verbosity_level > 0) {
        std::cout << "n_nonzero_approx: " << n_nonzero_approx << '\n';
        std::cout << '\n';
    }

    std::vector<Triplet> triplets_master;
    triplets_master.reserve(n_nonzero_approx);

    if (verbosity_level > 0) {
        std::cout << "IMAGE SPACE SIZE: " << i_max << " x " << j_max << " x " << k_max << '\n';
        std::cout << '\n';
    }

    // Used for bounds checks and flat indices only; no values are stored.
    Image img(0, 0, 0, i_max, j_max, k_max);

    if (verbosity_level > 0) {
        std::cout << "START PARALLEL LOOP" << '\n';
        std::cout << '\n';
    }

    ExceptionCollector errors;

#pragma omp parallel
    {
#pragma omp for
        for (int b = 0; b < block_size; b++) {
            errors.run([&] {
                if (verbosity_level > 1) {
                    std::cout << "OUTER LOOP " << b << '\n';
                    std::cout << '\n';
                }

                int i_sparse = b;
                std::vector<Triplet> triplets;

                for (int i = 0; i < sub_array_size - 1; i++) {
                    if (verbosity_level > 1) {
                        std::cout << "----> BLOCK " << b << ", DEBUG STEP " << i << " <----" << '\n';
                    }

                    float const dx = block.x[b][i + 1] - block.x[b][i];
                    float const dy = block.y[b][i + 1] - block.y[b][i];
                    float const dz = block.z[b][i + 1] - block.z[b][i];

                    int const i1 = voxel_index(block.y[b][i] - origin_y, slice_thickness);
                    int const j1 = voxel_index(origin_x - block.x[b][i], pixel_width);
                    int const k1 = voxel_index(block.z[b][i] - origin_z, pixel_width);

                    int const i2 = voxel_index(block.y[b][i + 1] - origin_y, slice_thickness);
                    int const j2 = voxel_index(origin_x - block.x[b][i + 1], pixel_width);
                    int const k2 = voxel_index(block.z[b][i + 1] - origin_z, pixel_width);

                    Voxel const start = Voxel(i1, j1, k1);
                    Voxel const end = Voxel(i2, j2, k2);

                    Voxel mid = Voxel(i1, j1, k1);
                    Voxel voxel_delta = end - start;

                    check_sampling_step(voxel_delta, i, b);

                    // Distances from each endpoint to the boundary plane crossed along each
                    // axis; only meaningful, and only used, where the voxel index changes.
                    float const dxb_1 =
                        distance_to_crossed_boundary(origin_x - block.x[b][i], j1, j2, pixel_width);
                    float const dyb_1 =
                        distance_to_crossed_boundary(block.y[b][i] - origin_y, i1, i2, slice_thickness);
                    float const dzb_1 =
                        distance_to_crossed_boundary(block.z[b][i] - origin_z, k1, k2, pixel_width);

                    float const dxb_2 =
                        distance_to_crossed_boundary(origin_x - block.x[b][i + 1], j1, j2, pixel_width);
                    float const dyb_2 =
                        distance_to_crossed_boundary(block.y[b][i + 1] - origin_y, i1, i2, slice_thickness);
                    float const dzb_2 =
                        distance_to_crossed_boundary(block.z[b][i + 1] - origin_z, k1, k2, pixel_width);

                    if (verbosity_level > 2) {
                        std::cout << '\n';
                        std::cout << "position" << '\n';
                        std::cout << "------------------------" << '\n';
                        std::cout << "r1: (" << block.x[b][i] << ", " << block.y[b][i] << ", "
                                  << block.z[b][i] << ")" << '\n';
                        std::cout << "r2: (" << block.x[b][i + 1] << ", " << block.y[b][i + 1] << ", "
                                  << block.z[b][i + 1] << ")" << '\n';
                        std::cout << "dr: (" << dx << ", " << dy << ", " << dz << ")" << '\n';
                        std::cout << '\n';

                        std::cout << "voxel values" << '\n';
                        std::cout << "------------------------" << '\n';
                        std::cout << "start: " << start << '\n';
                        std::cout << "end: " << end << '\n';
                        std::cout << "delta: " << voxel_delta << '\n';
                        std::cout << "mid: " << mid << '\n';
                        std::cout << '\n';

                        std::cout << "distances to boundaries" << '\n';
                        std::cout << "------------------------" << '\n';
                        std::cout << "dist_bound1: (" << dxb_1 << ", " << dyb_1 << ", " << dzb_1 << ")"
                                  << '\n';
                        std::cout << "dist_bound2: (" << dxb_2 << ", " << dyb_2 << ", " << dzb_2 << ")"
                                  << '\n';
                        std::cout << '\n';
                    }

                    if (!start.is_in(img) && !end.is_in(img) &&
                        (abs(voxel_delta.i()) + abs(voxel_delta.j()) + abs(voxel_delta.k()) < 2)) {
                        if (verbosity_level > 2) {
                            std::cout << '\n';
                            std::cout << "NOT IN IMAGE SPACE" << '\n';
                        }
                        continue;
                    }

                    // crossing no boundaries
                    if (voxel_delta.i() == 0 && voxel_delta.j() == 0 && voxel_delta.k() == 0) {
                        float const chord = distance(dx, dy, dz);

                        if (chord > 0.0) {
                            triplets.emplace_back(i_sparse, start.flat_index(img), chord);
                        }

                        if (verbosity_level > 2) {
                            std::cout << '\n';
                            std::cout << "NO BOUDARIES CROSSED" << '\n';
                            std::cout << "chord: " << distance(dx, dy, dz) << '\n';
                            std::cout << '\n';
                        }
                        continue;
                    }

                    float chord_in_pixel1 = NAN;
                    float chord_in_pixel2 = NAN;

                    float chord_x = std::sqrt(1 + ((dy / dx) * (dy / dx)) + ((dz / dx) * (dz / dx)));
                    float chord_y = std::sqrt(1 + ((dx / dy) * (dx / dy)) + ((dz / dy) * (dz / dy)));
                    float chord_z = std::sqrt(1 + ((dx / dz) * (dx / dz)) + ((dy / dz) * (dy / dz)));

                    if (!std::isfinite(chord_x)) {
                        chord_x = 0.0;
                    }

                    if (!std::isfinite(chord_y)) {
                        chord_y = 0.0;
                    }

                    if (!std::isfinite(chord_z)) {
                        chord_z = 0.0;
                    }

                    if (verbosity_level > 2) {
                        std::cout << '\n';
                        std::cout << "CHORDS" << '\n';
                        std::cout << "------------------------" << '\n';
                        std::cout << "c: (" << chord_x << ", " << chord_y << ", " << chord_z << ")" << '\n';
                        std::cout << '\n';
                    }

                    int const n_boundaries_crossed =
                        std::abs(voxel_delta.i()) + std::abs(voxel_delta.j()) + std::abs(voxel_delta.k());

                    // crossed 1 boundary
                    if (n_boundaries_crossed == 1) {

                        if (voxel_delta.i() != 0) {
                            chord_in_pixel1 = dyb_1 * chord_y;
                            chord_in_pixel2 = dyb_2 * chord_y;
                        } else if (voxel_delta.j() != 0) {
                            chord_in_pixel1 = dxb_1 * chord_x;
                            chord_in_pixel2 = dxb_2 * chord_x;
                        } else if (voxel_delta.k() != 0) {
                            chord_in_pixel1 = dzb_1 * chord_z;
                            chord_in_pixel2 = dzb_2 * chord_z;
                        }

                        if (start.is_in(img) && (chord_in_pixel1 > 0.0)) {
                            triplets.emplace_back(i_sparse, start.flat_index(img), chord_in_pixel1);
                        }

                        if (end.is_in(img) && (chord_in_pixel2 > 0.0)) {
                            triplets.emplace_back(i_sparse, end.flat_index(img), chord_in_pixel2);
                        }

                        if (verbosity_level > 2) {
                            std::cout << '\n';
                            std::cout << "CROSSED ONE BOUNDARY" << '\n';
                            std::cout << "------------------------" << '\n';
                            std::cout << "chord_in_pixel1: " << chord_in_pixel1 << '\n';
                            std::cout << "chord_in_pixel2: " << chord_in_pixel2 << '\n';
                        }
                        continue;
                    }

                    // crossing more than 1 boundary
                    float const p1_to_xb = dxb_1 * chord_x;
                    float const p1_to_yb = dyb_1 * chord_y;
                    float const p1_to_zb = dzb_1 * chord_z;

                    float const p2_to_xb = dxb_2 * chord_x;
                    float const p2_to_yb = dyb_2 * chord_y;
                    float const p2_to_zb = dzb_2 * chord_z;

                    if (verbosity_level > 2) {
                        std::cout << '\n';
                        std::cout << "p1_to_xb: " << p1_to_xb << '\n';
                        std::cout << "p1_to_yb: " << p1_to_yb << '\n';
                        std::cout << "p1_to_zb: " << p1_to_zb << '\n';

                        std::cout << "p2_to_xb: " << p2_to_xb << '\n';
                        std::cout << "p2_to_yb: " << p2_to_yb << '\n';
                        std::cout << "p2_to_zb: " << p2_to_zb << '\n';
                    }
                    if (voxel_delta.i() != 0 && voxel_delta.j() != 0 && voxel_delta.k() == 0) {

                        std::string condition;

                        bool fill_mid = true;

                        if (p1_to_yb < p1_to_xb) {
                            condition = "p1_to_yb < p1_to_xb";
                            chord_in_pixel1 = p1_to_yb;
                            chord_in_pixel2 = p2_to_xb;

                            mid += Voxel(voxel_delta.i(), 0, 0);
                        } else if (p1_to_xb < p1_to_yb) {
                            condition = "p1_to_xb < p1_to_yb";
                            chord_in_pixel1 = p1_to_xb;
                            chord_in_pixel2 = p2_to_yb;

                            mid += Voxel(0, voxel_delta.j(), 0);
                        } else {
                            condition = "p1_to_yb = p1_to_xb";
                            chord_in_pixel1 = p1_to_yb;
                            chord_in_pixel2 = p2_to_yb;

                            fill_mid = false;
                        }

                        if (verbosity_level > 2) {
                            float const yb_to_xb = std::abs(p1_to_yb - p1_to_xb);

                            std::cout << '\n';
                            std::cout << "CROSSED I AND J BOUNDARY: " << condition << '\n';
                            std::cout << "----------------------------------------------" << '\n';
                            std::cout << "distance from p1 to boundaries" << '\n';
                            std::cout << "  p1_to_xb: " << p1_to_xb << '\n';
                            std::cout << "  p1_to_yb: " << p1_to_yb << '\n';

                            std::cout << "distance from p2 to boundaries" << '\n';
                            std::cout << "  p2_to_xb: " << p2_to_xb << '\n';
                            std::cout << "  p2_to_yb: " << p2_to_yb << '\n';

                            std::cout << "distance between boundaries" << '\n';
                            std::cout << "  yb_to_xb: " << yb_to_xb << '\n';

                            std::cout << "distance in each pixel" << '\n';
                            std::cout << "  chord_in_pixel1: " << chord_in_pixel1 << '\n';
                            std::cout << "  chord_in_pixel2: " << chord_in_pixel2 << '\n';
                            std::cout << "  yb_to_xb: " << yb_to_xb << '\n';
                            std::cout << "  sum: " << chord_in_pixel1 + chord_in_pixel2 + yb_to_xb << '\n';

                            std::cout << "difference between sum and expected" << '\n';
                            std::cout << "  diff: "
                                      << (chord_in_pixel1 + chord_in_pixel2 + yb_to_xb) -
                                             std::sqrt(std::pow(dx, 2) + std::pow(dz, 2) + std::pow(dy, 2))
                                      << '\n';

                            std::cout << "inter pixel: " << mid << '\n';
                            std::cout << '\n';
                        }

                        if (fill_mid & mid.is_in(img)) {
                            float const yb_to_xb = std::abs(p1_to_yb - p1_to_xb);
                            triplets.emplace_back(i_sparse, mid.flat_index(img), yb_to_xb);
                        }
                    }

                    else if (voxel_delta.i() != 0 && voxel_delta.j() == 0 && voxel_delta.k() != 0) {
                        std::string condition;
                        bool fill_mid = true;

                        if (p1_to_yb < p1_to_zb) {
                            condition = "p1_to_yb < p1_to_zb";

                            chord_in_pixel1 = p1_to_yb;
                            chord_in_pixel2 = p2_to_zb;

                            mid += Voxel(voxel_delta.i(), 0, 0);
                        } else if (p1_to_zb < p1_to_yb) {
                            condition = "p1_to_zb < p1_to_yb";

                            chord_in_pixel1 = p1_to_zb;
                            chord_in_pixel2 = p2_to_yb;

                            mid += Voxel(0, 0, voxel_delta.k());
                        } else {
                            condition = "p1_to_yb = p1_to_zb";

                            chord_in_pixel1 = p1_to_zb;
                            chord_in_pixel2 = p2_to_zb;

                            fill_mid = false;
                        }

                        if (verbosity_level > 2) {
                            float const yb_to_zb = std::abs(p1_to_yb - p1_to_zb);

                            std::cout << '\n';
                            std::cout << "CROSSED I AND K BOUNDARY: " << condition << '\n';
                            std::cout << "----------------------------------------------" << '\n';

                            std::cout << "distance from p1 to boundaries" << '\n';
                            std::cout << "  p1_to_yb: " << p1_to_yb << '\n';
                            std::cout << "  p1_to_zb: " << p1_to_zb << '\n';

                            std::cout << "distance from p2 to boundaries" << '\n';
                            std::cout << "  p2_to_yb: " << p2_to_yb << '\n';
                            std::cout << "  p2_to_zb: " << p2_to_zb << '\n';

                            std::cout << "distance between boundaries" << '\n';
                            std::cout << "  yb_to_zb: " << yb_to_zb << '\n';

                            std::cout << "distance in each pixel" << '\n';
                            std::cout << "  chord_in_pixel1: " << chord_in_pixel1 << '\n';
                            std::cout << "  chord_in_pixel2: " << chord_in_pixel2 << '\n';
                            std::cout << "  yb_to_zb: " << yb_to_zb << '\n';
                            std::cout << "  sum: " << chord_in_pixel1 + chord_in_pixel2 + yb_to_zb << '\n';

                            std::cout << "difference between sum and expected" << '\n';
                            std::cout << "  diff: "
                                      << (chord_in_pixel1 + chord_in_pixel2 + yb_to_zb) -
                                             std::sqrt(std::pow(dx, 2) + std::pow(dz, 2) + std::pow(dy, 2))
                                      << '\n';

                            std::cout << "intermediate pixel: " << mid << '\n';
                            std::cout << '\n';
                        }

                        if (fill_mid & mid.is_in(img)) {
                            float const yb_to_zb = std::abs(p1_to_yb - p1_to_zb);
                            triplets.emplace_back(i_sparse, mid.flat_index(img), yb_to_zb);
                        }
                    }

                    else if (voxel_delta.i() == 0 && voxel_delta.j() != 0 && voxel_delta.k() != 0) {

                        std::string condition;
                        bool fill_mid = true;

                        if (p1_to_xb < p1_to_zb) {
                            condition = "p1_to_xb < p1_to_zb";

                            chord_in_pixel1 = p1_to_xb;
                            chord_in_pixel2 = p2_to_zb;

                            mid += Voxel(0, voxel_delta.j(), 0);
                        } else if (p1_to_zb < p1_to_xb) {
                            condition = "p1_to_zb < p1_to_xb";

                            chord_in_pixel1 = p1_to_zb;
                            chord_in_pixel2 = p2_to_xb;

                            mid += Voxel(0, 0, voxel_delta.k());
                        } else {
                            condition = "p1_to_xb = p1_to_zb";

                            chord_in_pixel1 = p1_to_xb;
                            chord_in_pixel2 = p2_to_xb;

                            fill_mid = false;
                        }

                        if (verbosity_level > 2) {
                            float xb_to_zb = std::abs(p1_to_xb - p1_to_zb);

                            std::cout << '\n';
                            std::cout << "CROSSED J AND K BOUNDARY: " << condition << '\n';
                            std::cout << "----------------------------------------------" << '\n';
                            std::cout << condition << '\n';
                            std::cout << "distance from p1 to boundaries" << '\n';
                            std::cout << "  p1_to_xb: " << p1_to_xb << '\n';
                            std::cout << "  p1_to_zb: " << p1_to_zb << '\n';

                            std::cout << "distance from p2 to boundaries" << '\n';
                            std::cout << "  p2_to_xb: " << p2_to_xb << '\n';
                            std::cout << "  p2_to_zb: " << p2_to_zb << '\n';

                            std::cout << "distance between boundaries" << '\n';
                            std::cout << "  xb_to_zb: " << xb_to_zb << '\n';

                            std::cout << "distance in each pixel" << '\n';
                            std::cout << "  chord_in_pixel1: " << chord_in_pixel1 << '\n';
                            std::cout << "  chord_in_pixel2: " << chord_in_pixel2 << '\n';
                            std::cout << "  xb_to_zb: " << xb_to_zb << '\n';
                            std::cout << "  sum: " << chord_in_pixel1 + chord_in_pixel2 + xb_to_zb << '\n';

                            std::cout << "difference between sum and expected" << '\n';
                            std::cout << "  diff: "
                                      << (chord_in_pixel1 + chord_in_pixel2 + xb_to_zb) -
                                             std::sqrt(std::pow(dx, 2) + std::pow(dz, 2) + std::pow(dy, 2))
                                      << '\n';

                            std::cout << "inter pixel: " << mid << '\n';
                            std::cout << '\n';
                        }

                        if (fill_mid & mid.is_in(img)) {
                            // distance between the two boundary intersections
                            float xb_to_zb = std::abs(p1_to_xb - p1_to_zb);
                            triplets.emplace_back(i_sparse, mid.flat_index(img), xb_to_zb);
                        }

                    }

                    // crossing 3 boundaries
                    else {
                        float const xb_to_yb = std::abs(p1_to_xb - p1_to_yb);
                        float const yb_to_zb = std::abs(p1_to_yb - p1_to_zb);
                        float const xb_to_zb = std::abs(p1_to_xb - p1_to_zb);

                        Voxel mid1 = start;
                        Voxel mid2 = start;

                        float mid1_value = 0.0;
                        float mid2_value = 0.0;

                        bool fill_mid = true;

                        std::string condition;
                        if ((p1_to_xb < p1_to_yb) & (p1_to_yb < p1_to_zb)) {
                            condition = "CONDITION 1: p1_to_xb < p1_to_yb < p1_to_zb";

                            chord_in_pixel1 = p1_to_xb;
                            chord_in_pixel2 = p2_to_zb;

                            mid1 += Voxel(0, voxel_delta.j(), 0);
                            mid2 += Voxel(voxel_delta.i(), voxel_delta.j(), 0);

                            mid1_value = xb_to_yb;
                            mid2_value = yb_to_zb;

                        } else if ((p1_to_xb < p1_to_zb) & (p1_to_zb < p1_to_yb)) {
                            condition = "CONDITION 2: p1_to_xb < p1_to_zb < p1_to_yb";

                            chord_in_pixel1 = p1_to_xb;
                            chord_in_pixel2 = p2_to_yb;

                            mid1 += Voxel(0, voxel_delta.j(), 0);
                            mid2 += Voxel(0, voxel_delta.j(), voxel_delta.k());

                            mid1_value = xb_to_zb;
                            mid2_value = yb_to_zb;

                        } else if ((p1_to_yb < p1_to_xb) & (p1_to_xb < p1_to_zb)) {
                            condition = "CONDITION 3: p1_to_yb < p1_to_xb < p1_to_zb";

                            chord_in_pixel1 = p1_to_yb;
                            chord_in_pixel2 = p2_to_zb;

                            mid1 += Voxel(voxel_delta.i(), 0, 0);
                            mid2 += Voxel(voxel_delta.i(), voxel_delta.j(), 0);

                            mid1_value = xb_to_yb;
                            mid2_value = xb_to_zb;

                        } else if ((p1_to_yb < p1_to_zb) & (p1_to_zb < p1_to_xb)) {
                            condition = "CONDITION 4: p1_to_yb < p1_to_zb < p1_to_xb";

                            chord_in_pixel1 = p1_to_yb;
                            chord_in_pixel2 = p2_to_xb;

                            mid1 += Voxel(voxel_delta.i(), 0, 0);
                            mid2 += Voxel(voxel_delta.i(), 0, voxel_delta.k());

                            mid1_value = yb_to_zb;
                            mid2_value = xb_to_zb;

                        } else if ((p1_to_zb < p1_to_xb) & (p1_to_xb < p1_to_yb)) {
                            condition = "CONDITION 5: p1_to_zb < p1_to_xb < p1_to_yb";

                            chord_in_pixel1 = p1_to_zb;
                            chord_in_pixel2 = p2_to_yb;

                            mid1 += Voxel(0, 0, voxel_delta.k());
                            mid2 += Voxel(0, voxel_delta.j(), voxel_delta.k());

                            mid1_value = xb_to_zb;
                            mid2_value = xb_to_yb;

                        } else if ((p1_to_zb < p1_to_yb) & (p1_to_yb < p1_to_xb)) {
                            condition = "CONDITION 6: p1_to_zb < p1_to_yb < p1_to_xb";

                            chord_in_pixel1 = p1_to_zb;
                            chord_in_pixel2 = p2_to_xb;

                            mid1 += Voxel(0, 0, voxel_delta.k());
                            mid2 += Voxel(voxel_delta.i(), 0, voxel_delta.k());

                            mid1_value = yb_to_zb;
                            mid2_value = xb_to_yb;

                        } else {
                            // Two of the distances are equal (the segment passes through an edge
                            // and crosses the third plane before or after it) or all three are
                            // (a corner). At most one intermediate voxel is filled: start shifted
                            // along every axis whose crossing comes first. mid2 stays at start
                            // with a zero chord and is not inserted.
                            float const p1_to_first = std::min({p1_to_xb, p1_to_yb, p1_to_zb});
                            float const p1_to_last = std::max({p1_to_xb, p1_to_yb, p1_to_zb});

                            chord_in_pixel1 = p1_to_first;
                            chord_in_pixel2 = (p1_to_last == p1_to_xb)   ? p2_to_xb
                                              : (p1_to_last == p1_to_yb) ? p2_to_yb
                                                                         : p2_to_zb;

                            if (p1_to_first < p1_to_last) {
                                condition = "CONDITION 7: two equal distances (edge)";

                                mid1 += Voxel(p1_to_yb == p1_to_first ? voxel_delta.i() : 0,
                                              p1_to_xb == p1_to_first ? voxel_delta.j() : 0,
                                              p1_to_zb == p1_to_first ? voxel_delta.k() : 0);
                                mid1_value = p1_to_last - p1_to_first;
                            } else {
                                condition = "CONDITION 8: three equal distances (corner)";

                                fill_mid = false;
                            }
                        }

                        if (verbosity_level > 2) {
                            std::cout << '\n';
                            std::cout << "mid1: " << mid1 << '\n';
                            std::cout << "mid2: " << mid2 << '\n';
                            std::cout << "mid1_values: " << mid1_value << '\n';
                            std::cout << "mid2_values: " << mid2_value << '\n';

                            std::cout << '\n';
                            std::cout << "CROSSED I, J, AND K BOUNDARY" << '\n';
                            std::cout << "------------------------" << '\n';
                            std::cout << condition << '\n';
                            std::cout << "distance from p1 to boundaries" << '\n';
                            std::cout << "  p1_to_xb: " << p1_to_xb << '\n';
                            std::cout << "  p1_to_yb: " << p1_to_yb << '\n';
                            std::cout << "  p1_to_zb: " << p1_to_zb << '\n';

                            std::cout << "distance from p2 to boundaries" << '\n';
                            std::cout << "  p2_to_xb: " << p2_to_xb << '\n';
                            std::cout << "  p2_to_yb: " << p2_to_yb << '\n';
                            std::cout << "  p2_to_zb: " << p2_to_zb << '\n';

                            std::cout << "distance between boundaries" << '\n';
                            std::cout << "  xb_to_yb: " << xb_to_yb << '\n';
                            std::cout << "  yb_to_zb: " << yb_to_zb << '\n';

                            std::cout << "distance in each pixel" << '\n';
                            std::cout << "  chord_in_pixel1: " << chord_in_pixel1 << '\n';
                            std::cout << "  chord_in_pixel2: " << chord_in_pixel2 << '\n';
                            std::cout << "  xb_to_yb: " << xb_to_yb << '\n';
                            std::cout << "  yb_to_zb: " << yb_to_zb << '\n';
                            std::cout << "  sum: " << chord_in_pixel1 + chord_in_pixel2 + xb_to_yb + yb_to_zb
                                      << '\n';

                            std::cout << "difference between sum and expected" << '\n';
                            std::cout << "  diff: "
                                      << (chord_in_pixel1 + chord_in_pixel2 + xb_to_yb + yb_to_zb) -
                                             std::sqrt(std::pow(dx, 2) + std::pow(dz, 2) + std::pow(dy, 2))
                                      << '\n';

                            std::cout << "inter pixel: " << mid << '\n';
                            std::cout << '\n';
                        }

                        if (fill_mid & mid1.is_in(img)) {
                            triplets.emplace_back(i_sparse, mid1.flat_index(img), mid1_value);
                        }

                        if (fill_mid & (mid2_value > 0.0F) & mid2.is_in(img)) {
                            triplets.emplace_back(i_sparse, mid2.flat_index(img), mid2_value);
                        }
                    }

                    if (start.is_in(img) && (chord_in_pixel1 > 0.0)) {
                        triplets.emplace_back(i_sparse, start.flat_index(img), chord_in_pixel1);
                    }

                    if (end.is_in(img) && (chord_in_pixel2 > 0.0)) {
                        triplets.emplace_back(i_sparse, end.flat_index(img), chord_in_pixel2);
                    }
                }

#pragma omp critical
                {
                    if (verbosity_level > 2) {
                        std::cout << '\n';
                        std::cout << "IN CRITICAL REGION" << '\n';
                    }
                    triplets_master.insert(triplets_master.end(), triplets.begin(), triplets.end());
                }
            });
        }
    }
    errors.rethrow();

    if (verbosity_level > 0) {
        std::cout << '\n';
        std::cout << "triplets_master.size(): " << triplets_master.size() << '\n';
    }

    return triplets_master;
}

void matrix::compute_and_store_sparse_matrix(
    std::string const &filename, std::vector<std::vector<float>> const &x_array,
    std::vector<std::vector<float>> const &y_array, std::vector<std::vector<float>> const &z_array,
    std::tuple<int, int, int> const img_shape, std::tuple<float, float, float> const coord_origin,
    float const pixel_width, float const slice_thickness, int const verbosity_level, int const n_threads,
    int const compression_level, int const shuffle_method, std::string const &codec) {
    compute_and_store_sparse_matrix(filename, make_block(x_array, y_array, z_array), img_shape, coord_origin,
                                    pixel_width, slice_thickness, verbosity_level, n_threads,
                                    compression_level, shuffle_method, codec);
}

void matrix::compute_and_store_sparse_matrix(std::string const &filename, TrajectoryBlock const &block,
                                             std::tuple<int, int, int> const img_shape,
                                             std::tuple<float, float, float> const coord_origin,
                                             float const pixel_width, float const slice_thickness,
                                             int const verbosity_level, int const n_threads,
                                             int const compression_level, int const shuffle_method,
                                             std::string const &codec) {
    std::vector<Triplet> triplets = compute_matrix_triplets(block, img_shape, coord_origin, pixel_width,
                                                            slice_thickness, verbosity_level, n_threads);

    int const block_size = block.size();
    int const n_cols = std::get<0>(img_shape) * std::get<1>(img_shape) * std::get<2>(img_shape);

    if (verbosity_level > 0) {
        std::cout << "create sparse matrix of size " << block_size << ", " << n_cols << '\n';
        std::cout << "from triplets of size " << triplets.size() << '\n';
    }

    SparseMatrix mat(block_size, n_cols);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    std::vector<Triplet>().swap(triplets); // no longer needed; release before compressing

    mat.makeCompressed();

    if (verbosity_level > 0) {
        std::cout << "nonzero matrix elements: " << mat.nonZeros() << '\n';
        std::cout << "sum of matrix elements: " << mat.sum() << '\n';
        std::cout << "marix will be stored in file '" << filename << "'" << '\n';
    }

    io::store_compressed_matrix(mat, filename, compression_level, shuffle_method, codec, verbosity_level);
}
