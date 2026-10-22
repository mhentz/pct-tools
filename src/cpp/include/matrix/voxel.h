#ifndef VOXEL_H
#define VOXEL_H

#include "grid.h"

#include <iostream>

/// Integer voxel index (i, j, k). The indices may lie outside the grid; is_in() tells whether
/// they do. i counts slices along y, j pixels along x and k pixels along z, see
/// exact_chord_matrix.cpp for the mapping from coordinates to indices.
class Voxel {
  private:
    int m_i, m_j, m_k;

  public:
    Voxel(int i, int j, int k) : m_i(i), m_j(j), m_k(k) { ; }

    [[nodiscard]] int i() const { return m_i; }
    [[nodiscard]] int j() const { return m_j; }
    [[nodiscard]] int k() const { return m_k; }

    void set_i(int i) { m_i = i; }
    void set_j(int j) { m_j = j; }
    void set_k(int k) { m_k = k; }

    /// Whether the voxel lies inside the half-open index ranges of `grid`.
    [[nodiscard]] bool is_in(Grid const &grid) const {
        bool is_inside_i = (grid.i_min() <= m_i) && (m_i < grid.i_max());
        bool is_inside_j = (grid.j_min() <= m_j) && (m_j < grid.j_max());
        bool is_inside_k = (grid.k_min() <= m_k) && (m_k < grid.k_max());
        return is_inside_i && is_inside_j && is_inside_k;
    }

    /// Row-major (C order) linear index i * j_max * k_max + j * k_max + k, used as the column
    /// index of the system matrix. Throws std::runtime_error when the index falls outside the
    /// grid, so callers should check is_in() first.
    [[nodiscard]] long long flat_index(int const &i_max, int const &j_max, int const &k_max) const {
        long long const index =
            (static_cast<long long>(m_i) * j_max * k_max) + (static_cast<long long>(m_j) * k_max) + m_k;
        long long const max_index = static_cast<long long>(i_max) * j_max * k_max;

        if ((index < 0) || (index >= max_index)) {
            std::cerr << "Index (" << m_i << " " << m_j << " " << m_k << ") -> " << index
                      << " out of bounds. Allowed index range is [0, " << max_index << ")." << '\n';
            throw std::runtime_error("Index out of bounds");
        }

        return index;
    }

    [[nodiscard]] long long flat_index(Grid const &grid) const {
        return flat_index(grid.i_max(), grid.j_max(), grid.k_max());
    }

    bool operator==(const Voxel &other) const {
        return (m_i == other.m_i) && (m_j == other.m_j) && (m_k == other.m_k);
    }

    bool operator!=(const Voxel &other) const { return !(*this == other); }

    Voxel operator-(const Voxel &other) const { return {m_i - other.i(), m_j - other.j(), m_k - other.k()}; }

    void operator+=(const Voxel &other) {
        m_i += other.i();
        m_j += other.j();
        m_k += other.k();
    }

    friend std::ostream &operator<<(std::ostream &os, const Voxel &voxel) {
        os << "(" << voxel.m_i << ", " << voxel.m_j << ", " << voxel.m_k << ")";
        return os;
    }
};

#endif /* VOXEL_H */
