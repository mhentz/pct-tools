// The same case-by-case tests as test_exact_chord_matrix.cpp, run through the sparse entry
// point matrix::compute_matrix_triplets() that the production path uses. Expected results
// are given as (row, column, value) triplets with the row-major flat voxel index; the two
// files should stay in step whenever a case is added to one of them.

#define CATCH_CONFIG_MAIN

#include "exact_chord_matrix.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinAbsMatcher;

void print_triplets(std::vector<Triplet> triplets, std::string name) {
    std::cout << std::setw(25) << name << ": ";
    for (auto triplet : triplets) {
        std::cout << triplet.row() << " " << triplet.col() << " " << triplet.value() << std::endl;
        if (triplets.size() > 1)
            std::cout << std::setw(28);
    }
    std::cout << std::endl;
}

bool triplet_comparison(Triplet a, Triplet b) {
    if (a.row() < b.row())
        return true;
    else if ((a.row() == b.row()) && (a.col() < b.col()))
        return true;
    else
        return false;
}

void compute_and_check_triplets(std::vector<float> x_values, std::vector<float> y_values,
                                std::vector<float> z_values, std::vector<Triplet> expected_triplets,
                                std::tuple<int, int, int> const img_shape,
                                std::tuple<float, float, float> const coord_origin, float const pixel_width,
                                float const slice_thickness, int verbosity_level = 0,
                                int const n_threads = 1) {
    // set up test arrays
    std::vector<std::vector<float>> x_array;
    std::vector<std::vector<float>> y_array;
    std::vector<std::vector<float>> z_array;

    x_array.push_back(x_values);
    y_array.push_back(y_values);
    z_array.push_back(z_values);

    // calculate triplets
    std::vector<Triplet> triplets =
        matrix::compute_matrix_triplets(x_array, y_array, z_array, img_shape, coord_origin, pixel_width,
                                        slice_thickness, verbosity_level, n_threads);

    std::vector<Triplet> non_zero_triplets;

    // remove any elements that are zero
    for (size_t i = 0; i < triplets.size(); i++) {
        if (triplets[i].value() > 1e-7) {
            if (verbosity_level > 0) {
                std::cout << "keeping (" << triplets[i].row() << " " << triplets[i].col() << " "
                          << triplets[i].value() << ")" << std::endl;
            }
            non_zero_triplets.push_back(triplets[i]);
        }
    }

    // sort triplets
    std::sort(non_zero_triplets.begin(), non_zero_triplets.end(), triplet_comparison);
    std::sort(expected_triplets.begin(), expected_triplets.end(), triplet_comparison);

    // print triplets
    if (verbosity_level > 0) {
        std::cout << std::endl;
        std::cout << "expected_triplets.size(): " << expected_triplets.size() << std::endl;
        print_triplets(expected_triplets, "expected_triplets");

        std::cout << "non_zero_triplets.size(): " << non_zero_triplets.size() << std::endl;
        print_triplets(non_zero_triplets, "actual_triplets");
    }

    // if (verbosity_level > 0) {
    //     std::cout << std::setw(0);
    //     std::cout << "triplets" << std::endl;
    //     for (size_t i = 0; i < triplets.size(); i++) {
    //         if (triplets[i].value() == 0)
    //             continue;
    //         std::cout << triplets[i].row() << " " << triplets[i].col() << " " << triplets[i].value()
    //                   << std::endl;
    //     }

    //     std::cout << "expected_triplets" << std::endl;
    //     for (int i = 0; i < expected_triplets.size(); i++) {
    //         std::cout << expected_triplets[i].row() << " " << expected_triplets[i].col() << " "
    //                   << expected_triplets[i].value() << std::endl;
    //     }
    // }

    REQUIRE((non_zero_triplets.size()) == expected_triplets.size());

    for (size_t i = 0; i < non_zero_triplets.size(); i++) {
        if (non_zero_triplets[i].value() == 0)
            continue;
        REQUIRE(non_zero_triplets[i].row() == expected_triplets[i].row());
        REQUIRE(non_zero_triplets[i].col() == expected_triplets[i].col());
        REQUIRE_THAT(non_zero_triplets[i].value(), WithinAbs(expected_triplets[i].value(), 1e-6));
    }
}

void compute_and_check_triplets_reverse(std::vector<float> x_array, std::vector<float> y_array,
                                        std::vector<float> z_array, std::vector<Triplet> expected_triplets,
                                        std::tuple<int, int, int> const img_shape,
                                        std::tuple<float, float, float> const coord_origin,
                                        float const pixel_width, float const slice_thickness,
                                        int verbosity_level = 0, int const n_threads = 1) {
    // reverse arrays
    std::reverse(x_array.begin(), x_array.end());
    std::reverse(y_array.begin(), y_array.end());
    std::reverse(z_array.begin(), z_array.end());

    if (verbosity_level > 0) {
        std::cout << "reversed x_array: ";
        for (size_t i = 0; i < x_array.size(); i++) {
            std::cout << x_array[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "reversed y_array: ";
        for (size_t i = 0; i < y_array.size(); i++) {
            std::cout << y_array[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "reversed z_array: ";
        for (size_t i = 0; i < z_array.size(); i++) {
            std::cout << z_array[i] << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }

    compute_and_check_triplets(x_array, y_array, z_array, expected_triplets, img_shape, coord_origin,
                               pixel_width, slice_thickness, verbosity_level, n_threads);
}

TEST_CASE("triplets can be calculated when no boundaries are crossed", "[matrix][parallel]") {
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
    int verbosity_level = 0;

    SECTION("crossing no boundary") {
        std::vector<float> x_values = {0.1, 0.7};
        std::vector<float> y_values = {0.05, 0.35};
        std::vector<float> z_values = {0.23, 0.43};

        std::vector<Triplet> expected_triplets = {Triplet(0, 2, 0.7)};

        compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape, coord_origin,
                                   pixel_width, slice_thickness, verbosity_level);

        compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);
    }
}

TEST_CASE("triplets can be calculated when one boundary is crossed", "[matrix][parallel]") {
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
    int verbosity_level = 0;

    SECTION("crossing boundary in i-direction") { // currently this corresponds to y-axis
        std::vector<float> x_values = {0.5, 0.5};
        std::vector<float> y_values = {0.75, 1.25};
        std::vector<float> z_values = {0.5, 0.5};

        std::vector<Triplet> expected_triplets = {Triplet(0, 2, 0.25), Triplet(0, 6, 0.25)};

        compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape, coord_origin,
                                   pixel_width, slice_thickness, verbosity_level);

        compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);
    }

    SECTION("crossing boundary in j-direction") { // currently x-axis
        std::vector<float> x_values = {0.75, 1.25};
        std::vector<float> y_values = {0.75, 0.75};
        std::vector<float> z_values = {0.5, 0.5};

        std::vector<Triplet> expected_triplets = {Triplet(0, 0, 0.25), Triplet(0, 2, 0.25)};

        compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape, coord_origin,
                                   pixel_width, slice_thickness, verbosity_level);

        compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);
    }

    SECTION("crossing boundary in k-direction") { // currently z-axis
        std::vector<float> x_values = {0.5, 0.5};
        std::vector<float> y_values = {1.5, 1.5};
        std::vector<float> z_values = {0.75, 1.25};

        std::vector<Triplet> expected_triplets = {Triplet(0, 6, 0.25), Triplet(0, 7, 0.25)};

        compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape, coord_origin,
                                   pixel_width, slice_thickness, verbosity_level);

        compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);
    }
}

TEST_CASE("triplets can be calculated when two boundaries are crossed", "[matrix][parallel]") {
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
    int verbosity_level = 0;

    SECTION("crossing boundaries in j- and k-direction") { // currently x- and z-axis
        SECTION("decreasing j, increasing k, order: j, k") {
            std::vector<float> x_values = {0.9, 1.2};
            std::vector<float> y_values = {0.5, 0.5};
            std::vector<float> z_values = {0.8, 1.1};

            std::vector<Triplet> expected_triplets = {Triplet(0, 0, 0.141421), Triplet(0, 1, 0.141421),
                                                      Triplet(0, 2, 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("decreasing j, increasing k, order: k, j") {
            std::vector<float> x_values = {0.8, 1.1};
            std::vector<float> y_values = {1.5, 1.5};
            std::vector<float> z_values = {0.9, 1.2};

            std::vector<Triplet> expected_triplets = {Triplet(0, 5, 0.141421), Triplet(0, 6, 0.141421),
                                                      Triplet(0, 7, 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("decreasing j, increasing k, via vertex") {
            std::vector<float> x_values = {0.8, 1.1};
            std::vector<float> y_values = {1.5, 1.5};
            std::vector<float> z_values = {0.8, 1.1};

            std::vector<Triplet> expected_triplets = {Triplet(0, 5, 0.141421), Triplet(0, 6, 2 * 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("increasing j, increasing k, order: k, j") {
            std::vector<float> x_values = {1.2, 0.9};
            std::vector<float> y_values = {0.5, 0.5};
            std::vector<float> z_values = {0.9, 1.2};

            std::vector<Triplet> expected_triplets = {Triplet(0, 0, 0.141421), Triplet(0, 1, 0.141421),
                                                      Triplet(0, 3, 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("increasing j, increasing k, order: j, k") {
            std::vector<float> x_values = {1.1, 0.8};
            std::vector<float> y_values = {1.5, 1.5};
            std::vector<float> z_values = {0.8, 1.1};

            std::vector<Triplet> expected_triplets = {Triplet(0, 4, 0.141421), Triplet(0, 6, 0.141421),
                                                      Triplet(0, 7, 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("increasing j, increasing k, via vertex") {
            std::vector<float> x_values = {1.1, 0.8};
            std::vector<float> y_values = {1.5, 1.5};
            std::vector<float> z_values = {0.9, 1.2};

            std::vector<Triplet> expected_triplets = {Triplet(0, 4, 0.141421), Triplet(0, 7, 2 * 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }
    }

    SECTION("crossing boundaries in i- and j-direction") { // currently y- and x-axis
        // section descriptions assume a top-down view onto the relevant axes
        // where i is the vertical direction and j is the horizontal direction

        SECTION("increasing i, decreasing j, order: i, j") {
            std::vector<float> x_values = {0.8, 1.1};
            std::vector<float> y_values = {0.9, 1.2};
            std::vector<float> z_values = {0.5, 0.5};

            std::vector<Triplet> expected_triplets = {Triplet(0, 2, 0.141421), Triplet(0, 4, 0.141421),
                                                      Triplet(0, 6, 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("increasing i, decreasing j, order: j, i") {
            std::vector<float> x_values = {0.9, 1.2};
            std::vector<float> y_values = {0.8, 1.1};
            std::vector<float> z_values = {0.5, 0.5};

            std::vector<Triplet> expected_triplets = {Triplet(0, 0, 0.141421), Triplet(0, 2, 0.141421),
                                                      Triplet(0, 4, 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("increasing i, decreasing j, via vertex") {
            std::vector<float> x_values = {0.8, 1.1};
            std::vector<float> y_values = {0.8, 1.1};
            std::vector<float> z_values = {1.5, 1.5};

            std::vector<Triplet> expected_triplets = {Triplet(0, 3, 2 * 0.141421), Triplet(0, 5, 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("decreasing i, decreasing j, order: j, i") {
            std::vector<float> x_values = {0.9, 1.2};
            std::vector<float> y_values = {1.2, 0.9};
            std::vector<float> z_values = {0.5, 0.5};

            std::vector<Triplet> expected_triplets = {Triplet(0, 0, 0.141421), Triplet(0, 4, 0.141421),
                                                      Triplet(0, 6, 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("decreasing i, decreasing j, order: i, j") {
            std::vector<float> x_values = {1.1, 0.8};
            std::vector<float> y_values = {0.8, 1.1};
            std::vector<float> z_values = {1.5, 1.5};

            std::vector<Triplet> expected_triplets = {Triplet(0, 1, 0.141421), Triplet(0, 3, 0.141421),
                                                      Triplet(0, 7, 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("decreasing i, decreasing j, via vertex") {
            std::vector<float> x_values = {0.9, 1.2};
            std::vector<float> y_values = {1.1, 0.8};
            std::vector<float> z_values = {1.5, 1.5};

            std::vector<Triplet> expected_triplets = {Triplet(0, 1, 2 * 0.141421), Triplet(0, 7, 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }
    }

    SECTION("crossing boundaries in i- and k-direction") { // currently y- and z-axis
        // section descriptions assume a top-down view onto the relevant axes
        // where i is the vertical direction and j is the horizontal direction

        SECTION("increasing i, increasing k, order: i, k") {
            std::vector<float> x_values = {0.5, 0.5};
            std::vector<float> y_values = {0.9, 1.2};
            std::vector<float> z_values = {0.8, 1.1};

            std::vector<Triplet> expected_triplets = {Triplet(0, 2, 0.141421), Triplet(0, 6, 0.141421),
                                                      Triplet(0, 7, 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("increasing i, increasing k, order: k, i") {
            std::vector<float> x_values = {0.5, 0.5};
            std::vector<float> y_values = {0.8, 1.1};
            std::vector<float> z_values = {0.9, 1.2};

            std::vector<Triplet> expected_triplets = {Triplet(0, 2, 0.141421), Triplet(0, 3, 0.141421),
                                                      Triplet(0, 7, 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("increasing i, increasing k, via vertex") {
            std::vector<float> x_values = {1.5, 1.5};
            std::vector<float> y_values = {0.8, 1.1};
            std::vector<float> z_values = {0.8, 1.1};

            std::vector<Triplet> expected_triplets = {Triplet(0, 0, 2 * 0.141421), Triplet(0, 5, 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("decreasing i, increasing k, order: k, i") {
            std::vector<float> x_values = {0.5, 0.5};
            std::vector<float> y_values = {1.2, 0.9};
            std::vector<float> z_values = {0.9, 1.2};

            std::vector<Triplet> expected_triplets = {Triplet(0, 3, 0.141421), Triplet(0, 6, 0.141421),
                                                      Triplet(0, 7, 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("decreasing i, increasing k, order: i, k") {
            std::vector<float> x_values = {1.5, 1.5};
            std::vector<float> y_values = {1.1, 0.8};
            std::vector<float> z_values = {0.8, 1.1};

            std::vector<Triplet> expected_triplets = {Triplet(0, 0, 0.141421), Triplet(0, 1, 0.141421),
                                                      Triplet(0, 4, 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("decreasing i, increasing k, via vertex") {
            std::vector<float> x_values = {1.5, 1.5};
            std::vector<float> y_values = {1.1, 0.8};
            std::vector<float> z_values = {0.9, 1.2};

            std::vector<Triplet> expected_triplets = {Triplet(0, 1, 2 * 0.141421), Triplet(0, 4, 0.141421)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }
    }
}

TEST_CASE("triplets can be calculated when three boundary are crossed", "[matrix][parallel]") {
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
    int verbosity_level = 0;

    SECTION("crossing boundary in i-, j- and k-direction") { // currently y-, x- and z-axis
        // because we are dealing with paths in 3D and using increments of 0.1 in each direction
        // the path in each voxel should be sqrt(3) * 0.1 = 0.173205

        SECTION("crossing boundaries in order: x, y, z") {
            std::vector<float> x_values = {0.9, 1.3};
            std::vector<float> y_values = {0.8, 1.2};
            std::vector<float> z_values = {0.7, 1.1};

            std::vector<Triplet> expected_triplets = {Triplet(0, 0, 0.173205), Triplet(0, 2, 0.173205),
                                                      Triplet(0, 4, 0.173205), Triplet(0, 5, 0.173205)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            // same as crossing in order z, y, x
            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("crossing boundaries in order: x, z, y") {
            std::vector<float> x_values = {0.9, 1.3};
            std::vector<float> y_values = {0.7, 1.1};
            std::vector<float> z_values = {0.8, 1.2};

            std::vector<Triplet> expected_triplets = {Triplet(0, 0, 0.173205), Triplet(0, 1, 0.173205),
                                                      Triplet(0, 2, 0.173205), Triplet(0, 5, 0.173205)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            // same as crossing in order y, z, x
            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("crossing boundaries in order: y, x, z") {
            std::vector<float> x_values = {0.8, 1.2};
            std::vector<float> y_values = {0.9, 1.3};
            std::vector<float> z_values = {0.7, 1.1};

            std::vector<Triplet> expected_triplets = {Triplet(0, 2, 0.173205), Triplet(0, 4, 0.173205),
                                                      Triplet(0, 5, 0.173205), Triplet(0, 6, 0.173205)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            // same as crossing in order z, x, y
            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("crossing boundaries in order: y, z, x") {
            std::vector<float> x_values = {0.7, 1.1};
            std::vector<float> y_values = {0.9, 1.3};
            std::vector<float> z_values = {0.8, 1.2};

            std::vector<Triplet> expected_triplets = {Triplet(0, 2, 0.173205), Triplet(0, 5, 0.173205),
                                                      Triplet(0, 6, 0.173205), Triplet(0, 7, 0.173205)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            // same as crossing in order x, z, y
            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("crossing boundaries in order: z, x, y") {
            std::vector<float> x_values = {0.8, 1.2};
            std::vector<float> y_values = {0.7, 1.1};
            std::vector<float> z_values = {0.9, 1.3};

            std::vector<Triplet> expected_triplets = {Triplet(0, 1, 0.173205), Triplet(0, 2, 0.173205),
                                                      Triplet(0, 3, 0.173205), Triplet(0, 5, 0.173205)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            // same as crossing in order y, x, z
            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("crossing boundaries in order: z, y, x") {
            std::vector<float> x_values = {0.7, 1.1};
            std::vector<float> y_values = {0.8, 1.2};
            std::vector<float> z_values = {0.9, 1.3};

            std::vector<Triplet> expected_triplets = {Triplet(0, 2, 0.173205), Triplet(0, 3, 0.173205),
                                                      Triplet(0, 5, 0.173205), Triplet(0, 7, 0.173205)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            // same as crossing in order x, y, z
            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("crossing boundaries at vertex") {
            std::vector<float> x_values = {0.9, 1.1};
            std::vector<float> y_values = {0.9, 1.1};
            std::vector<float> z_values = {0.9, 1.1};

            std::vector<Triplet> expected_triplets = {Triplet(0, 2, 0.173205), Triplet(0, 5, 0.173205)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }
    }
}

TEST_CASE("triplets can be calculated in edge cases", "[matrix][parallel]") {
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
    int verbosity_level = 0;

    SECTION("parallel to grid lines in i-direction") { // currently y-axis
        SECTION("segment lies just outside grid") {
            std::vector<float> x_values = {1.5, 1.5};
            std::vector<float> y_values = {-1.0001, -1.0001};
            std::vector<float> z_values = {0.25, 0.75};

            std::vector<Triplet> expected_triplets = {};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("segment lies exactly on grid edge") {
            std::vector<float> x_values = {1.5, 1.5};
            std::vector<float> y_values = {-1.0, -1.0};
            std::vector<float> z_values = {0.25, 0.75};

            std::vector<Triplet> expected_triplets = {Triplet(0, 0, 0.5)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("segment lies just above grid edge") {
            std::vector<float> x_values = {1.5, 1.5};
            std::vector<float> y_values = {0.0001, 0.0001};
            std::vector<float> z_values = {0.25, 0.75};

            std::vector<Triplet> expected_triplets = {Triplet(0, 4, 0.5)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }
    }

    SECTION("parallel to grid lines in j-direction") { // currently x-axis
        SECTION("segment lies just outside grid") {
            std::vector<float> x_values = {2.0001, 2.0001};
            std::vector<float> y_values = {0, 0};
            std::vector<float> z_values = {0.25, 0.75};

            std::vector<Triplet> expected_triplets = {};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("segment lies exactly on grid edge") {
            std::vector<float> x_values = {1.0, 1.0};
            std::vector<float> y_values = {-1.0, -1.0};
            std::vector<float> z_values = {0.25, 0.75};

            std::vector<Triplet> expected_triplets = {Triplet(0, 2, 0.5)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("segment lies just above grid edge") {
            std::vector<float> x_values = {0.9999, 0.9999};
            std::vector<float> y_values = {0.0001, 0.0001};
            std::vector<float> z_values = {0.25, 0.75};

            std::vector<Triplet> expected_triplets = {Triplet(0, 6, 0.5)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }
    }

    SECTION("parallel to grid lines in k-direction") { // currently z-axis
        SECTION("segment lies just outside grid") {
            std::vector<float> x_values = {0.25, 0.5};
            std::vector<float> y_values = {0.5, 0.5};
            std::vector<float> z_values = {-0.0001, -0.0001};

            std::vector<Triplet> expected_triplets = {};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("segment lies exactly on grid edge") {
            std::vector<float> x_values = {0.25, 0.5};
            std::vector<float> y_values = {0.5, 0.5};
            std::vector<float> z_values = {0.0, 0.0};

            std::vector<Triplet> expected_triplets = {Triplet(0, 6, 0.25)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("segment lies just above grid edge") {
            std::vector<float> x_values = {0.25, 0.5};
            std::vector<float> y_values = {0.5, 0.5};
            std::vector<float> z_values = {1.0001, 1.0001};

            std::vector<Triplet> expected_triplets = {Triplet(0, 7, 0.25)};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }
    }

    SECTION("paths outside grid") {
        SECTION("completely outside") {
            std::vector<float> x_values = {3, 4};
            std::vector<float> y_values = {-2, -3};
            std::vector<float> z_values = {5, 6};

            std::vector<Triplet> expected_triplets = {};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("partially outside crossing i-boundary") {
            SECTION("varying in i-direction") {
                std::vector<float> x_values = {1.5, 1.5};
                std::vector<float> y_values = {-1.25, -0.75};
                std::vector<float> z_values = {0.25, 0.25};

                std::vector<Triplet> expected_triplets = {Triplet(0, 0, 0.25)};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }

            SECTION("varying in i- and j-direction") {
                std::vector<float> x_values = {0.25, 0.5};
                std::vector<float> y_values = {-1.25, -0.75};
                std::vector<float> z_values = {0.25, 0.25};

                std::vector<Triplet> expected_triplets = {
                    Triplet(0, 2, std::sqrt(0.25 * 0.25 + 0.125 * 0.125))};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }

            SECTION("varying in i- and k-direction") {
                std::vector<float> x_values = {0.25, 0.25};
                std::vector<float> y_values = {-1.25, -0.75};
                std::vector<float> z_values = {0.0, 0.2};

                std::vector<Triplet> expected_triplets = {Triplet(0, 2, std::sqrt(0.25 * 0.25 + 0.1 * 0.1))};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }

            SECTION("varying in i-, j-, and k-direction") {
                std::vector<float> x_values = {0.2, 0.4};
                std::vector<float> y_values = {-1.25, -0.75};
                std::vector<float> z_values = {0.1, 0.5};

                std::vector<Triplet> expected_triplets = {
                    Triplet(0, 2, std::sqrt(0.25 * 0.25 + 0.2 * 0.2 + 0.1 * 0.1))};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }
        }

        SECTION("partially outside crossing j-boundary") {
            SECTION("varying in j-direction") {
                std::vector<float> x_values = {-0.25, 0.25};
                std::vector<float> y_values = {0.5, 0.5};
                std::vector<float> z_values = {0.25, 0.25};

                std::vector<Triplet> expected_triplets = {Triplet(0, 6, 0.25)};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }

            SECTION("varying in j-, k- direction") {
                std::vector<float> x_values = {1.8, 2.2};
                std::vector<float> y_values = {-0.75, -0.75};
                std::vector<float> z_values = {1.3, 1.5};

                std::vector<Triplet> expected_triplets = {Triplet(0, 1, std::sqrt(0.2 * 0.2 + 0.1 * 0.1))};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }

            SECTION("varying in i-, j-, and k-direction") {
                std::vector<float> x_values = {1.8, 2.2};
                std::vector<float> y_values = {-0.15, -0.75};
                std::vector<float> z_values = {1.3, 1.5};

                std::vector<Triplet> expected_triplets = {
                    Triplet(0, 1, std::sqrt(0.3 * 0.3 + 0.2 * 0.2 + 0.1 * 0.1))};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }
        }

        SECTION("partially outside crossing k-boundary") {
            SECTION("varying in k-direction") {
                std::vector<float> x_values = {0.5, 0.5};
                std::vector<float> y_values = {0.5, 0.5};
                std::vector<float> z_values = {1.75, 2.25};

                std::vector<Triplet> expected_triplets = {Triplet(0, 7, 0.25)};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }

            SECTION("varying in j-, k- direction") {
                std::vector<float> x_values = {1.6, 2.0};
                std::vector<float> y_values = {-0.75, -0.75};
                std::vector<float> z_values = {-0.1, 0.1};

                std::vector<Triplet> expected_triplets = {Triplet(0, 0, std::sqrt(0.2 * 0.2 + 0.1 * 0.1))};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }

            SECTION("varying in i-, j-, and k-direction") {
                std::vector<float> x_values = {0.6, 1.0};
                std::vector<float> y_values = {0.15, 0.75};
                std::vector<float> z_values = {-0.1, 0.1};

                std::vector<Triplet> expected_triplets = {
                    Triplet(0, 6, std::sqrt(0.3 * 0.3 + 0.2 * 0.2 + 0.1 * 0.1))};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }
        }

        SECTION("partially outside entering at vertex") {
            std::vector<float> x_values = {-0.2, 0.2};
            std::vector<float> y_values = {1.3, 0.7};
            std::vector<float> z_values = {-0.1, 0.1};

            std::vector<Triplet> expected_triplets = {
                Triplet(0, 6, std::sqrt(0.3 * 0.3 + 0.2 * 0.2 + 0.1 * 0.1))};

            compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                       coord_origin, pixel_width, slice_thickness, verbosity_level);

            compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                               coord_origin, pixel_width, slice_thickness, verbosity_level);
        }

        SECTION("one endpoint is inside, the other outside, intermediate pixel is outside") {
            SECTION("crossing i- and j-boundaries") {
                std::vector<float> x_values = {-0.1, 0.2};
                std::vector<float> y_values = {-1.25, -0.75};
                std::vector<float> z_values = {0.25, 0.25};

                std::vector<Triplet> expected_triplets = {
                    Triplet(0, 2, std::sqrt(0.25 * 0.25 + 0.15 * 0.15))};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }

            SECTION("crossing i- and k-boudaries") {
                std::vector<float> x_values = {0.25, 0.25};
                std::vector<float> y_values = {-1.25, -0.75};
                std::vector<float> z_values = {-0.1, 0.2};

                std::vector<Triplet> expected_triplets = {
                    Triplet(0, 2, std::sqrt(0.25 * 0.25 + 0.15 * 0.15))};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }

            SECTION("crossing j- and k-boudaries") {
                std::vector<float> x_values = {-0.25, 0.25};
                std::vector<float> y_values = {0.25, 0.25};
                std::vector<float> z_values = {-0.1, 0.2};

                std::vector<Triplet> expected_triplets = {
                    Triplet(0, 6, std::sqrt(0.25 * 0.25 + 0.15 * 0.15))};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }
        }

        SECTION("both endpoints are outside, intermediate pixel is inside") {
            SECTION("crossing i- and j-boundaries; 1") {
                std::vector<float> x_values = {-0.05, 0.15};
                std::vector<float> y_values = {0.85, 1.05};
                std::vector<float> z_values = {0.25, 0.25};

                std::vector<Triplet> expected_triplets = {Triplet(0, 6, std::sqrt(0.1 * 0.1 + 0.1 * 0.1))};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }

            SECTION("crossing i- and j-boundaries; 2") {
                std::vector<float> x_values = {1.85, 2.05};
                std::vector<float> y_values = {1.05, 0.85};
                std::vector<float> z_values = {0.25, 0.25};

                std::vector<Triplet> expected_triplets = {Triplet(0, 4, std::sqrt(0.1 * 0.1 + 0.1 * 0.1))};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }

            SECTION("crossing i- and k-boudaries; 1") {
                std::vector<float> x_values = {0.25, 0.25};
                std::vector<float> y_values = {0.85, 1.05};
                std::vector<float> z_values = {-0.05, 0.15};

                std::vector<Triplet> expected_triplets = {Triplet(0, 6, std::sqrt(0.1 * 0.1 + 0.1 * 0.1))};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }

            SECTION("crossing i- and k-boudaries; 2") {
                std::vector<float> x_values = {0.25, 0.25};
                std::vector<float> y_values = {1.05, 0.85};
                std::vector<float> z_values = {1.85, 2.05};

                std::vector<Triplet> expected_triplets = {Triplet(0, 7, std::sqrt(0.1 * 0.1 + 0.1 * 0.1))};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }

            SECTION("crossing j- and k-boudaries; 1") {
                std::vector<float> x_values = {1.85, 2.05};
                std::vector<float> y_values = {0.25, 0.25};
                std::vector<float> z_values = {-0.05, 0.15};

                std::vector<Triplet> expected_triplets = {Triplet(0, 4, std::sqrt(0.1 * 0.1 + 0.1 * 0.1))};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }

            SECTION("crossing j- and k-boudaries; 2") {
                std::vector<float> x_values = {2.05, 1.85};
                std::vector<float> y_values = {0.25, 0.25};
                std::vector<float> z_values = {1.85, 2.05};

                std::vector<Triplet> expected_triplets = {Triplet(0, 5, std::sqrt(0.1 * 0.1 + 0.1 * 0.1))};

                compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);

                compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                                   coord_origin, pixel_width, slice_thickness,
                                                   verbosity_level);
            }
        }
    }
}

TEST_CASE("triplets can be calculated with odd numbers of slices", "[matrix][parallel]") {
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
    int verbosity_level = 0;

    SECTION("in slice with index 0") {
        std::vector<float> x_values = {0.1, 0.2};
        std::vector<float> y_values = {-1.4, -1.1};
        std::vector<float> z_values = {0.7, 0.9};

        std::vector<Triplet> expected_triplets = {
            Triplet(0, 0, std::sqrt(0.3 * 0.3 + 0.2 * 0.2 + 0.1 * 0.1))};

        compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape, coord_origin,
                                   pixel_width, slice_thickness, verbosity_level);

        compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);
    }

    SECTION("in slice with index 1") {
        std::vector<float> x_values = {0.1, 0.2};
        std::vector<float> y_values = {-0.4, 0.2};
        std::vector<float> z_values = {0.7, 0.9};

        std::vector<Triplet> expected_triplets = {
            Triplet(0, 1, std::sqrt(0.6 * 0.6 + 0.2 * 0.2 + 0.1 * 0.1))};

        compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape, coord_origin,
                                   pixel_width, slice_thickness, verbosity_level);

        compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);
    }

    SECTION("in slice with index 2") {
        std::vector<float> x_values = {0.1, 0.2};
        std::vector<float> y_values = {0.6, 1.2};
        std::vector<float> z_values = {0.7, 0.9};

        std::vector<Triplet> expected_triplets = {
            Triplet(0, 2, std::sqrt(0.6 * 0.6 + 0.2 * 0.2 + 0.1 * 0.1))};

        compute_and_check_triplets(x_values, y_values, z_values, expected_triplets, img_shape, coord_origin,
                                   pixel_width, slice_thickness, verbosity_level);

        compute_and_check_triplets_reverse(x_values, y_values, z_values, expected_triplets, img_shape,
                                           coord_origin, pixel_width, slice_thickness, verbosity_level);
    }
}

// need test cases for more than 1 block
