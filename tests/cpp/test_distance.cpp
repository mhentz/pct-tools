#define CATCH_CONFIG_MAIN

#include "distance.h"

#include <algorithm>
#include <iomanip>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinAbsMatcher;

TEST_CASE("distance to the crossed boundary can be calculated", "[distance_to_crossed_boundary]") {
    // step from voxel 0 to voxel 1 of width 0.5: the boundary is at 0.5
    REQUIRE_THAT(distance_to_crossed_boundary(0.5, 0, 1, 0.5), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(distance_to_crossed_boundary(0.25, 0, 1, 0.5), WithinAbs(0.25, 1e-6));
    REQUIRE_THAT(distance_to_crossed_boundary(0.1, 0, 1, 0.5), WithinAbs(0.4, 1e-6));
    REQUIRE_THAT(distance_to_crossed_boundary(0.6, 0, 1, 0.5), WithinAbs(0.1, 1e-6));

    // the order of the indices does not matter
    REQUIRE_THAT(distance_to_crossed_boundary(0.6, 1, 0, 0.5), WithinAbs(0.1, 1e-6));

    // boundaries further from the origin, and a spacing that does not divide 1
    REQUIRE_THAT(distance_to_crossed_boundary(1.0, 2, 3, 0.3), WithinAbs(0.1, 1e-6));
    REQUIRE_THAT(distance_to_crossed_boundary(1.25, 3, 4, 0.3), WithinAbs(0.05, 1e-6));
    REQUIRE_THAT(distance_to_crossed_boundary(2.3, 1, 2, 2.0), WithinAbs(1.7, 1e-6));

    // negative displacements (points before the origin)
    REQUIRE_THAT(distance_to_crossed_boundary(-0.3, -1, 0, 1.0), WithinAbs(0.3, 1e-6));
    REQUIRE_THAT(distance_to_crossed_boundary(-1.2, -2, -1, 1.0), WithinAbs(0.2, 1e-6));
}

TEST_CASE("distance between two points can be calculated", "[distance]") {
    REQUIRE_THAT(distance(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(distance(0.0, 0.0, 0.0, 2.0, 3.0, 6.0), WithinAbs(7.0, 1e-6));
    REQUIRE_THAT(distance(2.0, 3.0, 6.0, 0.0, 0.0, 0.0), WithinAbs(7.0, 1e-6));
    REQUIRE_THAT(distance(-1.1, -1.4, -3.8, 0.9, 1.6, 2.2), WithinAbs(7.0, 1e-6));
    REQUIRE_THAT(distance(1.1, 1.4, 3.8, -0.9, -1.6, -2.2), WithinAbs(7.0, 1e-6));
    REQUIRE_THAT(distance(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), WithinAbs(5.1961524, 1e-6));
}

TEST_CASE("hypothenuse can be calculated based on three sides", "[distance]") {
    REQUIRE_THAT(distance(0.0, 0.0, 0.0), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(distance(2.0, 3.0, 6.0), WithinAbs(7.0, 1e-6));
    REQUIRE_THAT(distance(-2.0, -3.0, -6.0), WithinAbs(7.0, 1e-6));
    REQUIRE_THAT(distance(1.1, 1.4, 3.8), WithinAbs(4.1964271, 1e-6));
    REQUIRE_THAT(distance(3.0, 3.0, 3.0), WithinAbs(5.1961524, 1e-6));
}
