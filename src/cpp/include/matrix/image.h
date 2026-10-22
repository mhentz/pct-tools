#ifndef IMAGE_H
#define IMAGE_H

#include "grid.h"
#include "voxel.h"

#include <stdexcept>
#include <vector>

/// Dense 3-D grid of voxel values stored in row-major (C) order. The chord-length code uses
/// it as a plain Grid for bounds checks and flat indices; the value array is only allocated
/// by initialise_matrix(), and add_value() is not safe to call from several threads at once.
class Image : public Grid {
  private:
    int m_i_min, m_j_min, m_k_min, m_i_max, m_j_max, m_k_max;

    size_t m_size;
    std::vector<float> m_mat;

  public:
    Image(int i_min, int j_min, int k_min, int i_max, int j_max, int k_max)
        : m_i_min(i_min), m_j_min(j_min), m_k_min(k_min), m_i_max(i_max), m_j_max(j_max), m_k_max(k_max),
          m_size(static_cast<size_t>(m_i_max - m_i_min) * static_cast<size_t>(m_j_max - m_j_min) *
                 static_cast<size_t>(m_k_max - m_k_min)) {}
    void initialise_matrix() { m_mat.assign(m_size, 0.0F); }

    [[nodiscard]] bool is_initialised() const { return !m_mat.empty(); }

    /// Add `value` to the voxel's entry; voxels outside the grid are silently ignored.
    void add_value(Voxel const &voxel, float const value) {
        if (!is_initialised()) {
            throw std::runtime_error("The matrix has not been initialised. Please call "
                                     "Image::initialise_matrix() before setting values.");
        }
        if (voxel.is_in(*this)) {
            m_mat[voxel.flat_index(*this)] += value;
        }
    }

    [[nodiscard]] int i_min() const override { return m_i_min; }
    [[nodiscard]] int j_min() const override { return m_j_min; }
    [[nodiscard]] int k_min() const override { return m_k_min; }
    [[nodiscard]] int i_max() const override { return m_i_max; }
    [[nodiscard]] int j_max() const override { return m_j_max; }
    [[nodiscard]] int k_max() const override { return m_k_max; }
    [[nodiscard]] const float *mat() const override { return m_mat.data(); }
};

#endif /* IMAGE_H */
