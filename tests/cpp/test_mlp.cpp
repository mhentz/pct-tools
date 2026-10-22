// Most likely path checked against a direct numerical evaluation of the Schulte 2008 model.
//
// sigma^2 elements: E0^2 (1 + 0.038 ln(l / X0))^2 / X0 * integral_a^b w(u) / (beta^2 p^2)(u) du with
// w = (b - u)^2, (b - u), 1 for the t, t-theta and theta elements (Schulte et al. 2008, eqs. 7-9).
// The polynomial for 1 / (beta^2 p^2) is fitted per cm^n, hence the 10^-n rescaling to mm.

#include "mlp.h"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace {

constexpr double E0 = 13.6;  // MeV
constexpr double X0 = 361.0; // radiation length of water in mm
constexpr double HIGHLAND = 0.038;

std::vector<double> const GAMMEX{4.996e-6, 1.571e-7, -6.656e-9, 1.915e-9, -1.135e-10, 2.724e-12};
std::vector<double> const CTP528{7.512e-6, 3.238e-7, 1.380e-8, 1.339e-9, -1.332e-10, 9.354e-12};

double inv_beta2_p2(double u, std::vector<double> const &c) {
    double value = 0.0;
    for (size_t n = 0; n < c.size(); n++)
        value += c[n] / std::pow(10.0, static_cast<double>(n)) * std::pow(u, static_cast<double>(n));
    return value;
}

// Composite Simpson's rule; the integrands are smooth polynomials so this converges fast.
double simpson(std::function<double(double)> const &f, double a, double b, int n = 2000) {
    double const h = (b - a) / n;
    double sum = f(a) + f(b);
    for (int i = 1; i < n; i++)
        sum += (i % 2 == 1 ? 4.0 : 2.0) * f(a + i * h);
    return sum * h / 3.0;
}

struct Sigma {
    double t, t_theta, theta;
};

Sigma reference_sigma(double a, double b, std::vector<double> const &c, double highland = HIGHLAND) {
    double const prefactor = E0 * E0 * std::pow(1 + highland * std::log((b - a) / X0), 2) / X0;
    return {prefactor * simpson([&](double u) { return (b - u) * (b - u) * inv_beta2_p2(u, c); }, a, b),
            prefactor * simpson([&](double u) { return (b - u) * inv_beta2_p2(u, c); }, a, b),
            prefactor * simpson([&](double u) { return inv_beta2_p2(u, c); }, a, b)};
}

} // namespace

TEST_CASE("scattering matrix elements match numerical integration", "[mlp]") {
    for (auto const &c : {GAMMEX, CTP528}) {
        for (auto const [a, b] : {std::pair{0.0f, 7.0f}, std::pair{7.0f, 31.0f}, std::pair{31.0f, 200.0f}}) {
            Sigma const expected = reference_sigma(a, b, c);
            INFO("interval [" << a << ", " << b << "]");
            CHECK_THAT(mlp::compute_scattering_matrix_element_t(a, b, c),
                       Catch::Matchers::WithinRel(expected.t, 1e-6));
            CHECK_THAT(mlp::compute_scattering_matrix_element_cross_term(a, b, c),
                       Catch::Matchers::WithinRel(expected.t_theta, 1e-6));
            CHECK_THAT(mlp::compute_scattering_matrix_element_theta(a, b, c),
                       Catch::Matchers::WithinRel(expected.theta, 1e-6));
        }
    }
}

TEST_CASE("scattering matrix elements use the Lynch & Dahl constant 0.038, not 0.0038", "[mlp]") {
    // For l != X0 the logarithmic correction does not vanish, so a factor-ten typo is visible.
    float const a = 0.0f, b = 31.0f;
    Sigma const wrong = reference_sigma(a, b, GAMMEX, 0.0038);
    CHECK_THAT(mlp::compute_scattering_matrix_element_t(a, b, GAMMEX),
               !Catch::Matchers::WithinRel(wrong.t, 1e-3));
    CHECK_THAT(mlp::compute_scattering_matrix_element_theta(a, b, GAMMEX),
               !Catch::Matchers::WithinRel(wrong.theta, 1e-3));
}

TEST_CASE("built-in coefficients are looked up by name and unknown names throw", "[mlp]") {
    CHECK(mlp::get_coefficients("gammex") == GAMMEX);
    CHECK(mlp::get_coefficients("ctp528") == CTP528);
    CHECK(mlp::get_coefficients("spiral") == CTP528);
    CHECK_THROWS_AS(mlp::get_coefficients("water"), std::invalid_argument);
}

TEST_CASE("most likely path is a straight line for aligned entry and exit", "[mlp]") {
    std::vector<float> const z{0.0f, 50.0f, 100.0f, 150.0f, 200.0f};
    Eigen::VectorXf const y = mlp::compute_mlp(z, 1.0f, 0.0f, 1.0f, 0.0f, mlp::get_coefficients("gammex"), 1);
    for (int i = 0; i < y.size(); i++)
        CHECK_THAT(y[i], Catch::Matchers::WithinAbs(1.0, 1e-6));
}

TEST_CASE("most likely path matches the numerical reference for unequal segments", "[mlp]") {
    std::vector<float> const z{0.0f, 7.0f, 31.0f, 90.0f, 140.0f, 199.0f, 200.0f};
    float const h_in = 3.0f, theta_in = -0.02f, h_out = -1.5f, theta_out = 0.015f;

    for (auto const &c : {GAMMEX, CTP528}) {
        Eigen::VectorXf const y = mlp::compute_mlp(z, h_in, theta_in, h_out, theta_out, c, 1);
        Eigen::Vector2d const y_in(h_in, theta_in), y_out(h_out, theta_out);

        for (size_t i = 1; i + 1 < z.size(); i++) {
            auto inverse = [](Sigma const &s) {
                Eigen::Matrix2d m{{s.t, s.t_theta}, {s.t_theta, s.theta}};
                return m.inverse();
            };
            Eigen::Matrix2d const inv_s1 = inverse(reference_sigma(z[0], z[i], c));
            Eigen::Matrix2d const inv_s2 = inverse(reference_sigma(z[i], z.back(), c));
            Eigen::Matrix2d const r0{{1.0, static_cast<double>(z[i] - z[0])}, {0.0, 1.0}};
            Eigen::Matrix2d const r1{{1.0, static_cast<double>(z.back() - z[i])}, {0.0, 1.0}};
            Eigen::Vector2d const expected = (inv_s1 + r1.transpose() * inv_s2 * r1).inverse() *
                                             (inv_s1 * r0 * y_in + r1.transpose() * inv_s2 * y_out);
            INFO("z = " << z[i]);
            CHECK_THAT(y[static_cast<Eigen::Index>(i)], Catch::Matchers::WithinRel(expected(0), 1e-4));
        }
    }
}
