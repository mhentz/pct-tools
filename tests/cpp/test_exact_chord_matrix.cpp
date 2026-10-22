// Case-by-case tests of the dense entry point matrix::compute_matrix_elements().
//
// Each TEST_CASE covers one class of segment (no, one, two and three boundaries crossed), plus
// edge cases and an odd number of slices, against expected chord lengths worked out by hand
// for single two-point segments. These were the original tests of the algorithm;
// test_chord_reference.cpp now checks the same code against an independent reference
// implementation and is where new cases should go.

#define CATCH_CONFIG_MAIN

#include "exact_chord_matrix.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinAbsMatcher;

void print_array(std::vector<std::vector<std::vector<float>>> array, std::string name) {
    std::cout << std::setw(25) << name << ": ";

    for (size_t i = 0; i < array.size(); i++) {
        for (size_t j = 0; j < array[0].size(); j++) {
            for (size_t k = 0; k < array[0][0].size(); k++) {
                std::cout << array[i][j][k] << " ";
            }
        }
    }
    std::cout << std::endl;
}

void compute_and_check_matrix(std::vector<float> x_array, std::vector<float> y_array,
                              std::vector<float> z_array, std::vector<float> expected_values,
                              std::tuple<int, int, int> const &img_shape,
                              std::tuple<float, float, float> const &coord_origin, float const pixel_width,
                              float const slice_thickness, bool is_verbose) {
    // calculate matrix elements
    std::vector<std::vector<std::vector<float>>> matrix = matrix::compute_matrix_elements(
        x_array, y_array, z_array, img_shape, coord_origin, pixel_width, slice_thickness, is_verbose);

    if (is_verbose) {
        std::cout << "GOT MATRIX" << std::endl;
    }

    int const i_max = std::get<0>(img_shape);
    int const j_max = std::get<1>(img_shape);
    int const k_max = std::get<2>(img_shape);

    // set up expected matrix
    std::vector<std::vector<std::vector<float>>> expected_matrix(
        i_max, std::vector<std::vector<float>>(j_max, std::vector<float>(k_max, 0)));

    for (int i = 0; i < i_max; i++) {
        for (int j = 0; j < j_max; j++) {
            for (int k = 0; k < k_max; k++) {
                expected_matrix[i][j][k] = expected_values[i * j_max * k_max + j * k_max + k];
            }
        }
    }

    // compare expected and actual matrix
    int const n_expected_matrix = expected_matrix.size();
    int const n_matrix = matrix.size();

    REQUIRE(n_expected_matrix == n_matrix);

    if (is_verbose) {
        print_array(expected_matrix, "expected_matrix_array");
        print_array(matrix, "matrix_array");
    }

    for (int i = 0; i < i_max; i++) {
        for (int j = 0; j < j_max; j++) {
            for (int k = 0; k < k_max; k++) {
                REQUIRE_THAT(expected_matrix[i][j][k], WithinAbs(matrix[i][j][k], 1e-6));
            }
        }
    }
}

void compute_and_check_matrix_reverse(std::vector<float> x_array, std::vector<float> y_array,
                                      std::vector<float> z_array, std::vector<float> expected_values,
                                      std::tuple<int, int, int> const &img_shape,
                                      std::tuple<float, float, float> const &coord_origin,
                                      float const pixel_width, float const slice_thickness, bool is_verbose) {
    // reverse arrays
    std::reverse(x_array.begin(), x_array.end());
    std::reverse(y_array.begin(), y_array.end());
    std::reverse(z_array.begin(), z_array.end());

    compute_and_check_matrix(x_array, y_array, z_array, expected_values, img_shape, coord_origin, pixel_width,
                             slice_thickness, is_verbose);
}

TEST_CASE("matrix elements can be calculated when no boundaries are crossed", "[matrix][single]") {
    // set up parameters for matrix:
    //    - 2x2x2 image with 1x1x1 mm voxels, basically a stack of 2x2 images
    //    - uses tensor indexing:
    //        - i-dimension: indexes slices in the stack
    //        - j-dimension: indexes rows in the slices (i-dimension in 2D matrices)
    //        - k-dimension: indexes cols in the slices (j-dimension in 2D matrices)
    //    - sets origin to the maximum value in the j-direction to mirror typical image
    //       or matrix indexing

    auto img_shape = std::make_tuple(2, 2, 2);
    auto coord_origin = std::make_tuple(2, 0, 0);
    float pixel_width = 1.0;
    float slice_thickness = 1.0;
    bool is_verbose = false;

    SECTION("crossing no boundary") {
        std::vector<float> x_values = {0.1, 0.7};
        std::vector<float> y_values = {0.05, 0.35};
        std::vector<float> z_values = {0.23, 0.43};

        std::vector<float> expected_values = {0, 0, 0.7, 0, 0, 0, 0, 0};

        compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                 pixel_width, slice_thickness, is_verbose);

        compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);
    }
}

TEST_CASE("matrix elements can be calculated when one boundary is crossed", "[matrix][single]") {
    // set up parameters for matrix:
    //    - 2x2x2 image with 1x1x1 mm voxels, basically a stack of 2x2 images
    //    - uses tensor indexing:
    //        - i-dimension: indexes slices in the stack
    //        - j-dimension: indexes rows in the slices (i-dimension in 2D matrices)
    //        - k-dimension: indexes cols in the slices (j-dimension in 2D matrices)
    //    - sets origin to the maximum value in the j-direction to mirror typical image
    //       or matrix indexing

    auto img_shape = std::make_tuple(2, 2, 2);
    auto coord_origin = std::make_tuple(2, 0, 0);
    float pixel_width = 1.0;
    float slice_thickness = 1.0;
    bool is_verbose = false;

    SECTION("crossing boundary in i-direction") { // currently this corresponds to y-axis
        std::vector<float> x_values = {0.5, 0.5};
        std::vector<float> y_values = {0.75, 1.25};
        std::vector<float> z_values = {0.5, 0.5};

        std::vector<float> expected_values = {0, 0, 0.25, 0, 0, 0, 0.25, 0};

        compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                 pixel_width, slice_thickness, is_verbose);

        compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);
    }

    SECTION("crossing boundary in j-direction") { // currently x-axis
        std::vector<float> x_values = {0.75, 1.25};
        std::vector<float> y_values = {0.75, 0.75};
        std::vector<float> z_values = {0.5, 0.5};

        std::vector<float> expected_values = {0.25, 0, 0.25, 0, 0, 0, 0, 0};

        compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                 pixel_width, slice_thickness, is_verbose);

        compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);
    }

    SECTION("crossing boundary in k-direction") { // currently z-axis
        std::vector<float> x_values = {0.5, 0.5};
        std::vector<float> y_values = {1.5, 1.5};
        std::vector<float> z_values = {0.75, 1.25};

        std::vector<float> expected_values = {0, 0, 0, 0, 0, 0, 0.25, 0.25};

        compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                 pixel_width, slice_thickness, is_verbose);

        compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);
    }
}

TEST_CASE("matrix elements can be calculated when two boundaries are crossed", "[matrix][single]") {
    // set up parameters for matrix:
    //    - 2x2x2 image with 1x1x1 mm voxels, basically a stack of 2x2 images
    //    - uses tensor indexing:
    //        - i-dimension: indexes slices in the stack
    //        - j-dimension: indexes rows in the slices (i-dimension in 2D matrices)
    //        - k-dimension: indexes cols in the slices (j-dimension in 2D matrices)
    //    - sets origin to the maximum value in the j-direction to mirror typical image
    //       or matrix indexing

    auto img_shape = std::make_tuple(2, 2, 2);
    auto coord_origin = std::make_tuple(2, 0, 0);
    float pixel_width = 1.0;
    float slice_thickness = 1.0;
    bool is_verbose = false;

    SECTION("crossing boundaries in j- and k-direction") { // currently x- and z-axis
        SECTION("decreasing j, increasing k, order: j, k") {
            std::vector<float> x_values = {0.9, 1.2};
            std::vector<float> y_values = {0.5, 0.5};
            std::vector<float> z_values = {0.8, 1.1};

            std::vector<float> expected_values = {0.141421, 0.141421, 0.141421, 0, 0, 0, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("decreasing j, increasing k, order: k, j") {
            std::vector<float> x_values = {0.8, 1.1};
            std::vector<float> y_values = {1.5, 1.5};
            std::vector<float> z_values = {0.9, 1.2};

            std::vector<float> expected_values = {0, 0, 0, 0, 0, 0.141421, 0.141421, 0.141421};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("decreasing j, increasing k, via vertex") {
            std::vector<float> x_values = {0.8, 1.1};
            std::vector<float> y_values = {1.5, 1.5};
            std::vector<float> z_values = {0.8, 1.1};

            std::vector<float> expected_values = {0, 0, 0, 0, 0, 0.141421, 2 * 0.141421, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("increasing j, increasing k, order: k, j") {
            std::vector<float> x_values = {1.2, 0.9};
            std::vector<float> y_values = {0.5, 0.5};
            std::vector<float> z_values = {0.9, 1.2};

            std::vector<float> expected_values = {0.141421, 0.141421, 0, 0.141421, 0, 0, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("increasing j, increasing k, order: j, k") {
            std::vector<float> x_values = {1.1, 0.8};
            std::vector<float> y_values = {1.5, 1.5};
            std::vector<float> z_values = {0.8, 1.1};

            std::vector<float> expected_values = {0, 0, 0, 0, 0.141421, 0, 0.141421, 0.141421};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("increasing j, increasing k, via vertex") {
            std::vector<float> x_values = {1.1, 0.8};
            std::vector<float> y_values = {1.5, 1.5};
            std::vector<float> z_values = {0.9, 1.2};

            std::vector<float> expected_values = {0, 0, 0, 0, 0.141421, 0, 0, 2 * 0.141421};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }
    }

    SECTION("crossing boundaries in i- and j-direction") { // currently y- and x-axis
        // section descriptions assume a top-down view onto the relevant axes
        // where i is the vertical direction and j is the horizontal direction

        SECTION("increasing i, decreasing j, order: i, j") {
            std::vector<float> x_values = {0.8, 1.1};
            std::vector<float> y_values = {0.9, 1.2};
            std::vector<float> z_values = {0.5, 0.5};

            std::vector<float> expected_values = {0, 0, 0.141421, 0, 0.141421, 0, 0.141421, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("increasing i, decreasing j, order: j, i") {
            std::vector<float> x_values = {0.9, 1.2};
            std::vector<float> y_values = {0.8, 1.1};
            std::vector<float> z_values = {0.5, 0.5};

            std::vector<float> expected_values = {0.141421, 0, 0.141421, 0, 0.141421, 0, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("increasing i, decreasing j, via vertex") {
            std::vector<float> x_values = {0.8, 1.1};
            std::vector<float> y_values = {0.8, 1.1};
            std::vector<float> z_values = {1.5, 1.5};

            std::vector<float> expected_values = {0, 0, 0, 2 * 0.141421, 0, 0.141421, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("decreasing i, decreasing j, order: j, i") {
            std::vector<float> x_values = {0.9, 1.2};
            std::vector<float> y_values = {1.2, 0.9};
            std::vector<float> z_values = {0.5, 0.5};

            std::vector<float> expected_values = {0.141421, 0, 0, 0, 0.141421, 0, 0.141421, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("decreasing i, decreasing j, order: i, j") {
            std::vector<float> x_values = {1.1, 0.8};
            std::vector<float> y_values = {0.8, 1.1};
            std::vector<float> z_values = {1.5, 1.5};

            std::vector<float> expected_values = {0, 0.141421, 0, 0.141421, 0, 0, 0, 0.141421};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("decreasing i, decreasing j, via vertex") {
            std::vector<float> x_values = {0.9, 1.2};
            std::vector<float> y_values = {1.1, 0.8};
            std::vector<float> z_values = {1.5, 1.5};

            std::vector<float> expected_values = {0, 2 * 0.141421, 0, 0, 0, 0, 0, 0.141421};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }
    }

    SECTION("crossing boundaries in i- and k-direction") { // currently y- and z-axis
        // section descriptions assume a top-down view onto the relevant axes
        // where i is the vertical direction and j is the horizontal direction

        SECTION("increasing i, increasing k, order: i, k") {
            std::vector<float> x_values = {0.5, 0.5};
            std::vector<float> y_values = {0.9, 1.2};
            std::vector<float> z_values = {0.8, 1.1};

            std::vector<float> expected_values = {0, 0, 0.141421, 0, 0, 0, 0.141421, 0.141421};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("increasing i, increasing k, order: k, i") {
            std::vector<float> x_values = {0.5, 0.5};
            std::vector<float> y_values = {0.8, 1.1};
            std::vector<float> z_values = {0.9, 1.2};

            std::vector<float> expected_values = {0, 0, 0.141421, 0.141421, 0, 0, 0, 0.141421};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("increasing i, increasing k, via vertex") {
            std::vector<float> x_values = {1.5, 1.5};
            std::vector<float> y_values = {0.8, 1.1};
            std::vector<float> z_values = {0.8, 1.1};

            std::vector<float> expected_values = {2 * 0.141421, 0, 0, 0, 0, 0.141421, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("decreasing i, increasing k, order: k, i") {
            std::vector<float> x_values = {0.5, 0.5};
            std::vector<float> y_values = {1.2, 0.9};
            std::vector<float> z_values = {0.9, 1.2};

            std::vector<float> expected_values = {0, 0, 0, 0.141421, 0, 0, 0.141421, 0.141421};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("decreasing i, increasing k, order: i, k") {
            std::vector<float> x_values = {1.5, 1.5};
            std::vector<float> y_values = {1.1, 0.8};
            std::vector<float> z_values = {0.8, 1.1};

            std::vector<float> expected_values = {0.141421, 0.141421, 0, 0, 0.141421, 0, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("decreasing i, increasing k, via vertex") {
            std::vector<float> x_values = {1.5, 1.5};
            std::vector<float> y_values = {1.1, 0.8};
            std::vector<float> z_values = {0.9, 1.2};

            std::vector<float> expected_values = {0, 2 * 0.141421, 0, 0, 0.141421, 0, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }
    }
}

TEST_CASE("matrix elements can be calculated when three boundary are crossed", "[matrix][single]") {
    // set up parameters for matrix:
    //    - 2x2x2 image with 1x1x1 mm voxels, basically a stack of 2x2 images
    //    - uses tensor indexing:
    //        - i-dimension: indexes slices in the stack
    //        - j-dimension: indexes rows in the slices (i-dimension in 2D matrices)
    //        - k-dimension: indexes cols in the slices (j-dimension in 2D matrices)
    //    - sets origin to the maximum value in the j-direction to mirror typical image
    //       or matrix indexing

    auto img_shape = std::make_tuple(2, 2, 2);
    auto coord_origin = std::make_tuple(2, 0, 0);
    float pixel_width = 1.0;
    float slice_thickness = 1.0;
    bool is_verbose = false;

    SECTION("crossing boundary in i-, j- and k-direction") { // currently y-, x- and z-axis
        // because we are dealing with paths in 3D and using increments of 0.1 in each direction
        // the path in each voxel should be sqrt(3) * 0.1 = 0.173205

        SECTION("crossing boundaries in order: x, y, z") {
            std::vector<float> x_values = {0.9, 1.3};
            std::vector<float> y_values = {0.8, 1.2};
            std::vector<float> z_values = {0.7, 1.1};

            std::vector<float> expected_values = {0.173205, 0, 0.173205, 0, 0.173205, 0.173205, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            // same as crossing in order z, y, x
            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("crossing boundaries in order: x, z, y") {
            std::vector<float> x_values = {0.9, 1.3};
            std::vector<float> y_values = {0.7, 1.1};
            std::vector<float> z_values = {0.8, 1.2};

            std::vector<float> expected_values = {0.173205, 0.173205, 0.173205, 0, 0, 0.173205, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            // same as crossing in order y, z, x
            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("crossing boundaries in order: y, x, z") {
            std::vector<float> x_values = {0.8, 1.2};
            std::vector<float> y_values = {0.9, 1.3};
            std::vector<float> z_values = {0.7, 1.1};

            std::vector<float> expected_values = {0, 0, 0.173205, 0, 0.173205, 0.173205, 0.173205, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            // same as crossing in order z, x, y
            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("crossing boundaries in order: y, z, x") {
            std::vector<float> x_values = {0.7, 1.1};
            std::vector<float> y_values = {0.9, 1.3};
            std::vector<float> z_values = {0.8, 1.2};

            std::vector<float> expected_values = {0, 0, 0.173205, 0, 0, 0.173205, 0.173205, 0.173205};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            // same as crossing in order x, z, y
            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("crossing boundaries in order: z, x, y") {
            std::vector<float> x_values = {0.8, 1.2};
            std::vector<float> y_values = {0.7, 1.1};
            std::vector<float> z_values = {0.9, 1.3};

            std::vector<float> expected_values = {0, 0.173205, 0.173205, 0.173205, 0, 0.173205, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            // same as crossing in order y, x, z
            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("crossing boundaries in order: z, y, x") {
            std::vector<float> x_values = {0.7, 1.1};
            std::vector<float> y_values = {0.8, 1.2};
            std::vector<float> z_values = {0.9, 1.3};

            std::vector<float> expected_values = {0, 0, 0.173205, 0.173205, 0, 0.173205, 0, 0.173205};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            // same as crossing in order x, y, z
            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("crossing boundaries at vertex") {
            std::vector<float> x_values = {0.9, 1.1};
            std::vector<float> y_values = {0.9, 1.1};
            std::vector<float> z_values = {0.9, 1.1};

            std::vector<float> expected_values = {0, 0, 0.173205, 0, 0, 0.173205, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }
    }
}

TEST_CASE("matrix elements can be calculated in edge cases", "[matrix][single]") {
    // edge cases exclude crossing at a vertex because that is included in the tests above

    // included edge cases:
    //   - segments that lie entirely on an axis
    //   - some of the path is outside the image
    //   - the entire path is outside the image

    // set up parameters for matrix:
    //    - 3x2x2 image with 1x1x1 mm voxels, basically a stack of 3 2x2 images
    //    - uses tensor indexing:
    //        - i-dimension: indexes slices in the stack
    //        - j-dimension: indexes rows in the slices (i-dimension in 2D matrices)
    //        - k-dimension: indexes cols in the slices (j-dimension in 2D matrices)
    //    - sets origin to the maximum value in the j-direction to mirror typical image
    //       or matrix indexing

    auto img_shape = std::make_tuple(2, 2, 2);
    auto coord_origin = std::make_tuple(2, -1, 0);
    float pixel_width = 1.0;
    float slice_thickness = 1.0;
    bool is_verbose = false;

    SECTION("parallel to grid lines in i-direction") { // currently y-axis
        SECTION("segment lies just outside grid") {
            std::vector<float> x_values = {1.5, 1.5};
            std::vector<float> y_values = {-1.0001, -1.0001};
            std::vector<float> z_values = {0.25, 0.75};

            std::vector<float> expected_values = {0, 0, 0, 0, 0, 0, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("segment lies exactly on grid edge") {
            std::vector<float> x_values = {1.5, 1.5};
            std::vector<float> y_values = {-1.0, -1.0};
            std::vector<float> z_values = {0.25, 0.75};

            std::vector<float> expected_values = {0.5, 0, 0, 0, 0, 0, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("segment lies just above grid edge") {
            std::vector<float> x_values = {1.5, 1.5};
            std::vector<float> y_values = {0.0001, 0.0001};
            std::vector<float> z_values = {0.25, 0.75};

            std::vector<float> expected_values = {0, 0, 0, 0, 0.5, 0, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }
    }

    SECTION("parallel to grid lines in j-direction") { // currently x-axis
        SECTION("segment lies just outside grid") {
            std::vector<float> x_values = {2.0001, 2.0001};
            std::vector<float> y_values = {0, 0};
            std::vector<float> z_values = {0.25, 0.75};

            std::vector<float> expected_values = {0, 0, 0, 0, 0, 0, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("segment lies exactly on grid edge") {
            std::vector<float> x_values = {1.0, 1.0};
            std::vector<float> y_values = {-1.0, -1.0};
            std::vector<float> z_values = {0.25, 0.75};

            std::vector<float> expected_values = {0, 0, 0.5, 0, 0, 0, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("segment lies just above grid edge") {
            std::vector<float> x_values = {0.9999, 0.9999};
            std::vector<float> y_values = {0.0001, 0.0001};
            std::vector<float> z_values = {0.25, 0.75};

            std::vector<float> expected_values = {0, 0, 0, 0, 0, 0, 0.5, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }
    }

    SECTION("parallel to grid lines in k-direction") { // currently z-axis
        SECTION("segment lies just outside grid") {
            std::vector<float> x_values = {0.25, 0.5};
            std::vector<float> y_values = {0.5, 0.5};
            std::vector<float> z_values = {-0.0001, -0.0001};

            std::vector<float> expected_values = {0, 0, 0, 0, 0, 0, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("segment lies exactly on grid edge") {
            std::vector<float> x_values = {0.25, 0.5};
            std::vector<float> y_values = {0.5, 0.5};
            std::vector<float> z_values = {0.0, 0.0};

            std::vector<float> expected_values = {0, 0, 0, 0, 0, 0, 0.25, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("segment lies just above grid edge") {
            std::vector<float> x_values = {0.25, 0.5};
            std::vector<float> y_values = {0.5, 0.5};
            std::vector<float> z_values = {1.0001, 1.0001};

            std::vector<float> expected_values = {0, 0, 0, 0, 0, 0, 0, 0.25};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }
    }

    SECTION("paths outside grid") {
        SECTION("completely outside") {
            std::vector<float> x_values = {3, 4};
            std::vector<float> y_values = {-2, -3};
            std::vector<float> z_values = {5, 6};

            std::vector<float> expected_values = {0, 0, 0, 0, 0, 0, 0, 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("partially outside crossing i-boundary") {
            SECTION("varying in i-direction") {
                std::vector<float> x_values = {1.5, 1.5};
                std::vector<float> y_values = {-1.25, -0.75};
                std::vector<float> z_values = {0.25, 0.25};

                std::vector<float> expected_values = {0.25, 0, 0, 0, 0, 0, 0, 0};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }

            SECTION("varying in i- and j-direction") {
                std::vector<float> x_values = {0.25, 0.5};
                std::vector<float> y_values = {-1.25, -0.75};
                std::vector<float> z_values = {0.25, 0.25};

                std::vector<float> expected_values = {
                    0, 0, (float)std::sqrt(0.25 * 0.25 + 0.125 * 0.125), 0, 0, 0, 0, 0};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }

            SECTION("varying in i-, j-, and k-direction") {
                std::vector<float> x_values = {0.2, 0.4};
                std::vector<float> y_values = {-1.25, -0.75};
                std::vector<float> z_values = {0.1, 0.5};

                std::vector<float> expected_values = {
                    0, 0, (float)std::sqrt(0.25 * 0.25 + 0.2 * 0.2 + 0.1 * 0.1), 0, 0, 0, 0, 0};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }
        }

        SECTION("partially outside crossing j-boundary") {
            SECTION("varying in j-direction") {
                std::vector<float> x_values = {-0.25, 0.25};
                std::vector<float> y_values = {0.5, 0.5};
                std::vector<float> z_values = {0.25, 0.25};

                std::vector<float> expected_values = {0, 0, 0, 0, 0, 0, 0.25, 0};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }

            SECTION("varying in j-, k- direction") {
                std::vector<float> x_values = {1.8, 2.2};
                std::vector<float> y_values = {-0.75, -0.75};
                std::vector<float> z_values = {1.3, 1.5};

                std::vector<float> expected_values = {
                    0, (float)std::sqrt(0.2 * 0.2 + 0.1 * 0.1), 0, 0, 0, 0, 0, 0};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }

            SECTION("varying in i-, j-, and k-direction") {
                std::vector<float> x_values = {1.8, 2.2};
                std::vector<float> y_values = {-0.15, -0.75};
                std::vector<float> z_values = {1.3, 1.5};

                std::vector<float> expected_values = {
                    0, (float)std::sqrt(0.3 * 0.3 + 0.2 * 0.2 + 0.1 * 0.1), 0, 0, 0, 0, 0, 0};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }
        }

        SECTION("partially outside crossing k-boundary") {
            SECTION("varying in k-direction") {
                std::vector<float> x_values = {0.5, 0.5};
                std::vector<float> y_values = {0.5, 0.5};
                std::vector<float> z_values = {1.75, 2.25};

                std::vector<float> expected_values = {0, 0, 0, 0, 0, 0, 0, 0.25};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }

            SECTION("varying in j-, k- direction") {
                std::vector<float> x_values = {1.6, 2.0};
                std::vector<float> y_values = {-0.75, -0.75};
                std::vector<float> z_values = {-0.1, 0.1};

                std::vector<float> expected_values = {
                    (float)std::sqrt(0.2 * 0.2 + 0.1 * 0.1), 0, 0, 0, 0, 0, 0, 0};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }

            SECTION("varying in i-, j-, and k-direction") {
                std::vector<float> x_values = {0.6, 1.0};
                std::vector<float> y_values = {0.15, 0.75};
                std::vector<float> z_values = {-0.1, 0.1};

                std::vector<float> expected_values = {
                    0, 0, 0, 0, 0, 0, (float)std::sqrt(0.3 * 0.3 + 0.2 * 0.2 + 0.1 * 0.1), 0};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }
        }

        SECTION("partially outside entering at vertex") {
            std::vector<float> x_values = {-0.2, 0.2};
            std::vector<float> y_values = {1.3, 0.7};
            std::vector<float> z_values = {-0.1, 0.1};

            std::vector<float> expected_values = {
                0, 0, 0, 0, 0, 0, (float)std::sqrt(0.3 * 0.3 + 0.2 * 0.2 + 0.1 * 0.1), 0};

            compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                     pixel_width, slice_thickness, is_verbose);

            compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                             coord_origin, pixel_width, slice_thickness, is_verbose);
        }

        SECTION("one endpoint is inside, the other outside, intermediate pixel is outside") {
            SECTION("crossing i- and j-boundaries") {
                std::vector<float> x_values = {-0.1, 0.2};
                std::vector<float> y_values = {-1.25, -0.75};
                std::vector<float> z_values = {0.25, 0.25};

                std::vector<float> expected_values = {
                    0, 0, (float)std::sqrt(0.25 * 0.25 + 0.15 * 0.15), 0, 0, 0, 0, 0};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }

            SECTION("crossing i- and k-boudaries") {
                std::vector<float> x_values = {0.25, 0.25};
                std::vector<float> y_values = {-1.25, -0.75};
                std::vector<float> z_values = {-0.1, 0.2};

                std::vector<float> expected_values = {
                    0, 0, (float)std::sqrt(0.25 * 0.25 + 0.15 * 0.15), 0, 0, 0, 0, 0};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }

            SECTION("crossing j- and k-boudaries") {
                std::vector<float> x_values = {-0.25, 0.25};
                std::vector<float> y_values = {0.25, 0.25};
                std::vector<float> z_values = {-0.1, 0.2};

                std::vector<float> expected_values = {
                    0, 0, 0, 0, 0, 0, (float)std::sqrt(0.25 * 0.25 + 0.15 * 0.15), 0};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }
        }

        SECTION("both endpoints are outside, intermediate pixel is inside") {
            SECTION("crossing i- and j-boundaries; 1") {
                std::vector<float> x_values = {-0.05, 0.15};
                std::vector<float> y_values = {0.85, 1.05};
                std::vector<float> z_values = {0.25, 0.25};

                std::vector<float> expected_values = {
                    0, 0, 0, 0, 0, 0, (float)std::sqrt(0.1 * 0.1 + 0.1 * 0.1), 0};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }

            SECTION("crossing i- and j-boundaries; 2") {
                std::vector<float> x_values = {1.85, 2.05};
                std::vector<float> y_values = {1.05, 0.85};
                std::vector<float> z_values = {0.25, 0.25};

                std::vector<float> expected_values = {0, 0, 0, 0, (float)std::sqrt(0.1 * 0.1 + 0.1 * 0.1),
                                                      0, 0, 0};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }

            SECTION("crossing i- and k-boudaries; 1") {
                std::vector<float> x_values = {0.25, 0.25};
                std::vector<float> y_values = {0.85, 1.05};
                std::vector<float> z_values = {-0.05, 0.15};

                std::vector<float> expected_values = {
                    0, 0, 0, 0, 0, 0, (float)std::sqrt(0.1 * 0.1 + 0.1 * 0.1), 0};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }

            SECTION("crossing i- and k-boudaries; 2") {
                std::vector<float> x_values = {0.25, 0.25};
                std::vector<float> y_values = {1.05, 0.85};
                std::vector<float> z_values = {1.85, 2.05};

                std::vector<float> expected_values = {0, 0, 0, 0,
                                                      0, 0, 0, (float)std::sqrt(0.1 * 0.1 + 0.1 * 0.1)};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }

            SECTION("crossing j- and k-boudaries; 1") {
                std::vector<float> x_values = {1.85, 2.05};
                std::vector<float> y_values = {0.25, 0.25};
                std::vector<float> z_values = {-0.05, 0.15};

                std::vector<float> expected_values = {0, 0, 0, 0, (float)std::sqrt(0.1 * 0.1 + 0.1 * 0.1),
                                                      0, 0, 0};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }

            SECTION("crossing j- and k-boudaries; 2") {
                std::vector<float> x_values = {2.05, 1.85};
                std::vector<float> y_values = {0.25, 0.25};
                std::vector<float> z_values = {1.85, 2.05};

                std::vector<Triplet> expected_triplets = {Triplet(0, 5, std::sqrt(0.1 * 0.1 + 0.1 * 0.1))};
                std::vector<float> expected_values = {0, 0, 0, 0, 0, (float)std::sqrt(0.1 * 0.1 + 0.1 * 0.1),
                                                      0, 0};

                compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);

                compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                                 coord_origin, pixel_width, slice_thickness, is_verbose);
            }
        }
    }
}

TEST_CASE("matrix elements can be calculated with odd numbers of slices", "[matrix][single]") {
    // tests 0.5 offset for i-direction when we have odd number of slices

    // set up parameters for matrix:
    //    - 2x2x2 image with 1x1x1 mm voxels, basically a stack of 2x2 images
    //    - uses tensor indexing:
    //        - i-dimension: indexes slices in the stack
    //        - j-dimension: indexes rows in the slices (i-dimension in 2D matrices)
    //        - k-dimension: indexes cols in the slices (j-dimension in 2D matrices)
    //    - sets origin to the maximum value in the j-direction to mirror typical image
    //       or matrix indexing

    auto img_shape = std::make_tuple(3, 1, 1);
    auto coord_origin = std::make_tuple(1, -1.5, 0);
    float pixel_width = 1.0;
    float slice_thickness = 1.0;
    bool is_verbose = false;

    SECTION("in slice with index 0") {
        std::vector<float> x_values = {0.1, 0.2};
        std::vector<float> y_values = {-1.4, -1.1};
        std::vector<float> z_values = {0.7, 0.9};

        std::vector<float> expected_values = {(float)std::sqrt(0.3 * 0.3 + 0.2 * 0.2 + 0.1 * 0.1), 0, 0};

        compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                 pixel_width, slice_thickness, is_verbose);

        compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);
    }

    SECTION("in slice with index 1") {
        std::vector<float> x_values = {0.1, 0.2};
        std::vector<float> y_values = {-0.4, 0.2};
        std::vector<float> z_values = {0.7, 0.9};

        std::vector<float> expected_values = {0, (float)std::sqrt(0.6 * 0.6 + 0.2 * 0.2 + 0.1 * 0.1), 0};

        compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                 pixel_width, slice_thickness, is_verbose);

        compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);
    }

    SECTION("in slice with index 2") {
        std::vector<float> x_values = {0.1, 0.2};
        std::vector<float> y_values = {0.6, 1.2};
        std::vector<float> z_values = {0.7, 0.9};

        std::vector<float> expected_values = {0, 0, (float)std::sqrt(0.6 * 0.6 + 0.2 * 0.2 + 0.1 * 0.1)};

        compute_and_check_matrix(x_values, y_values, z_values, expected_values, img_shape, coord_origin,
                                 pixel_width, slice_thickness, is_verbose);

        compute_and_check_matrix_reverse(x_values, y_values, z_values, expected_values, img_shape,
                                         coord_origin, pixel_width, slice_thickness, is_verbose);
    }
}
