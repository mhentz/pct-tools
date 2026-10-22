#ifndef DISTANCE_H
#define DISTANCE_H

/// Distance along one axis from a point to the voxel boundary crossed by a step between the
/// voxels of indices `index_1` and `index_2`, which must differ by at most one. `displacement`
/// is the point's signed displacement from the grid origin along that axis, the same quantity
/// voxel_index() takes, so the boundary lies at max(index_1, index_2) * spacing. The value is
/// meaningless when the two indices are equal. Takes doubles so that a displacement formed
/// from float coordinates is not rounded again before the subtraction: the chord factor
/// multiplies that rounding by up to 1 / slope for segments nearly parallel to the plane.
float distance_to_crossed_boundary(double displacement, int index_1, int index_2, double spacing);

/// Index of the voxel a point falls into along one axis, given its signed displacement from
/// the grid origin and the voxel size along that axis. Takes doubles so that positions
/// computed in double precision are not rounded before the floor.
int voxel_index(double displacement, double spacing);

/// Euclidean distance between points A and B.
float distance(float a_x, float a_y, float a_z, float b_x, float b_y, float b_z);
/// Euclidean norm of the vector (delta_x, delta_y, delta_z).
float distance(float delta_x, float delta_y, float delta_z);

#endif /* DISTANCE_H */
