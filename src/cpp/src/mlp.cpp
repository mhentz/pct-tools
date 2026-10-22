#include "mlp.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include <Eigen/Dense>

// Scattering-power model of Schulte et al. 2008, Med. Phys. 35, 4849 (after Williams 2004 and
// Lynch & Dahl 1991): sigma^2 = E0^2 (1 + 0.038 ln(l / X0))^2 / X0 * integral of 1/(beta^2 p^2),
// with E0 = 13.6 MeV, X0 = 361 mm (water) and 1/(beta^2 p^2) fitted by a 5th-order polynomial
// whose coefficients are given per cm^n and rescaled to mm below.
namespace {

constexpr double E0 = 13.6;          // MeV
constexpr double X0 = 361.0;         // radiation length of water in mm
constexpr double LYNCH_DAHL = 0.038; // constant of the logarithmic correction

/// E0^2 (1 + 0.038 ln(l / X0))^2 / X0 for a path length l = b - a.
double scattering_prefactor(float const a, float const b) {
    double const highland = E0 * (1 + (LYNCH_DAHL * std::log((b - a) / X0)));
    return highland * highland / X0;
}

} // namespace

// Coefficients a_0 .. a_5 of the 5th-order polynomial fit to the mean 1/(beta^2 p^2) of
// protons as a function of depth, in c^2/MeV^2 per cm^n, following the procedure of Schulte
// et al. 2008, Sec. II.C (fit to the mean value recorded at regular depth intervals in a
// Geant4 simulation). The sets below are our own fits to the Geant4 simulations of the
// phantoms in protonct-sim, one per phantom size and beam energy: the CTP528 and spiral
// phantoms share a size and were both simulated at 200 MeV, so they share a fit; the Gammex
// phantom was simulated at 250 MeV and has its own. They are not the published values for a
// 20 cm water cube at 200 MeV (Schulte's Table I: a_0 = 7.457e-6, a_1 = 4.548e-7, ...).
std::vector<double> mlp::get_coefficients(std::string const &phantom_type) {
    if (phantom_type == "gammex") { // 250 MeV
        return {4.996e-6, 1.571e-7, -6.656e-9, 1.915e-9, -1.135e-10, 2.724e-12};
    }
    if (phantom_type == "ctp528" || phantom_type == "spiral") { // 200 MeV
        return {7.512e-6, 3.238e-7, 1.380e-8, 1.339e-9, -1.332e-10, 9.354e-12};
    }
    throw std::invalid_argument("Unknown phantom type '" + phantom_type +
                                "'; expected 'ctp528', 'spiral' or 'gammex'");
}

double mlp::compute_scattering_matrix_element_t(float const &a, float const &b,
                                                std::vector<double> const &coefficients) {
    double integral = 0.0;

    for (size_t idx = 0; idx < coefficients.size(); idx++) {
        auto const n = static_cast<double>(idx);
        double element_1 = std::pow(b, 3 + n) * ((1 / (3 + n)) - (2 / (2 + n)) + (1 / (1 + n)));
        double element_2 = std::pow(a, n) * ((std::pow(a, 3) / (3 + n)) - (2 * b * std::pow(a, 2) / (2 + n)) +
                                             (std::pow(b, 2) * a / (1 + n)));
        integral += coefficients[idx] / std::pow(10, n) * (element_1 - element_2);
    }
    return scattering_prefactor(a, b) * integral;
}

double mlp::compute_scattering_matrix_element_theta(float const &a, float const &b,
                                                    std::vector<double> const &coefficients) {
    double integral = 0.0;

    for (size_t idx = 0; idx < coefficients.size(); idx++) {
        auto const n = static_cast<double>(idx);
        integral += coefficients[idx] / std::pow(10, n) * (std::pow(b, n + 1) - std::pow(a, n + 1)) / (n + 1);
    }
    return scattering_prefactor(a, b) * integral;
}

double mlp::compute_scattering_matrix_element_cross_term(float const &a, float const &b,
                                                         std::vector<double> const &coefficients) {
    double integral = 0.0;

    for (size_t idx = 0; idx < coefficients.size(); idx++) {
        auto const n = static_cast<double>(idx);
        double element_1 = std::pow(b, n + 2) / ((n + 1) * (n + 2));
        double element_2 = (b * std::pow(a, n + 1) / (n + 1)) - (std::pow(a, n + 2) / (n + 2));
        integral += coefficients[idx] / std::pow(10, n) * (element_1 - element_2);
    }
    return scattering_prefactor(a, b) * integral;
}

Eigen::Matrix2d mlp::construct_inverse_sigma_matrix(float const &a, float const &b,
                                                    std::vector<double> const &coefficients) {
    double sigma_t = mlp::compute_scattering_matrix_element_t(a, b, coefficients);
    double sigma_t_theta = mlp::compute_scattering_matrix_element_cross_term(a, b, coefficients);
    double sigma_theta = mlp::compute_scattering_matrix_element_theta(a, b, coefficients);

    double determinant = (sigma_t * sigma_theta) - std::pow(sigma_t_theta, 2);

    if (determinant == 0) {
        determinant = 1e-30; // avoid singular matrix
    }

    Eigen::Matrix2d inverse_sigma_matrix{
        {sigma_theta / determinant, -sigma_t_theta / determinant},
        {-sigma_t_theta / determinant, sigma_t / determinant},
    };

    return inverse_sigma_matrix;
}

Eigen::Matrix2d mlp::construct_rotation_matrix(float const &a, float const &b) {
    Eigen::Matrix2d mat{{1, a - b}, {0, 1}};
    return mat;
}

Eigen::VectorXf mlp::compute_mlp(std::vector<float> const &z, float const &h_in, float const &theta_in,
                                 float const &h_out, float const &theta_out,
                                 std::vector<double> const &coefficients, int const n_threads) {
    omp_set_num_threads(n_threads);

    auto const n_z = static_cast<int>(z.size());

    Eigen::VectorXf y(n_z);
    y(0) = h_in;
    y(n_z - 1) = h_out;

    Eigen::Vector2d y_in(h_in, theta_in);
    Eigen::Vector2d y_out(h_out, theta_out);

    float z_in = z[0];
    float z_out = z[n_z - 1];

#pragma omp parallel for
    for (int i = 1; i < n_z - 1; i++) {
        Eigen::Matrix2d inv_sigma_1 = mlp::construct_inverse_sigma_matrix(z_in, z[i], coefficients);
        Eigen::Matrix2d inv_sigma_2 = mlp::construct_inverse_sigma_matrix(z[i], z_out, coefficients);

        Eigen::Matrix2d r_0 = mlp::construct_rotation_matrix(z[i], z[0]);
        Eigen::Matrix2d r_1 = mlp::construct_rotation_matrix(z[n_z - 1], z[i]);

        Eigen::Matrix2d lhs = (inv_sigma_1 + r_1.transpose() * inv_sigma_2 * r_1).inverse();
        Eigen::Vector2d rhs = (inv_sigma_1 * r_0 * y_in + r_1.transpose() * inv_sigma_2 * y_out);

        Eigen::Vector2d y_i = lhs * rhs;
        y(i) = static_cast<float>(y_i(0));
    }

    return y;
}
