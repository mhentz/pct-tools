#include "distance.h"

#include <algorithm>
#include <cmath>

float distance_to_crossed_boundary(double const displacement, int const index_1, int const index_2,
                                   double const spacing) {
    // The boundary between voxels m and m + 1 lies at (m + 1) * spacing; with the two indices
    // differing by at most one, that is max(index_1, index_2) * spacing.
    double const boundary = static_cast<double>(std::max(index_1, index_2)) * spacing;
    return static_cast<float>(std::abs(displacement - boundary));
}

int voxel_index(double const displacement, double const spacing) {
    return static_cast<int>(std::floor(displacement / spacing));
}

float distance(float a_x, float a_y, float a_z, float b_x, float b_y, float b_z) {
    // Calculate distance between points A and B
    return distance(b_x - a_x, b_y - a_y, b_z - a_z);
}

float distance(float delta_x, float delta_y, float delta_z) {
    // Calculate length of hypothenuse based on three sides
    return std::sqrt((delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z));
}
