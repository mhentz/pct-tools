#ifndef GRID_H
#define GRID_H

/// Interface of a 3-D voxel grid: half-open index ranges [i_min, i_max) etc. and read access
/// to the values. Implemented by Image; Voxel only depends on this interface.
class Grid {
  public:
    Grid() = default;
    Grid(Grid const &) = default;
    Grid(Grid &&) = default;
    Grid &operator=(Grid const &) = default;
    Grid &operator=(Grid &&) = default;
    virtual ~Grid() = default;

    [[nodiscard]] virtual int i_min() const = 0;
    [[nodiscard]] virtual int j_min() const = 0;
    [[nodiscard]] virtual int k_min() const = 0;
    [[nodiscard]] virtual int i_max() const = 0;
    [[nodiscard]] virtual int j_max() const = 0;
    [[nodiscard]] virtual int k_max() const = 0;
    [[nodiscard]] virtual const float *mat() const = 0;
};

#endif /* GRID_H */
