#ifndef MLP_H
#define MLP_H

#include <string>
#include <vector>

#include <Eigen/Dense>

/// Most likely path (MLP) of a proton after Schulte et al. 2008, Med. Phys. 35, 4849.
///
/// The state of a proton at depth z is (t, theta): transverse position and direction. Multiple
/// Coulomb scattering between depths a and b is described by the 2x2 covariance matrix Sigma
/// whose elements integrate 1 / (beta^2 p^2) over depth with weights (b - u)^2, (b - u) and 1.
/// The integrand is approximated by a 5th-order polynomial in depth given by six coefficients
/// c_n per cm^n (rescaled to mm internally). Depths and positions are in mm, angles in rad.
namespace mlp {

/// Built-in polynomial coefficients fitted for a named phantom: "ctp528" and "spiral" (same
/// size, both at 200 MeV, hence one shared fit) or "gammex" (250 MeV). Throws
/// std::invalid_argument for any other name. The coefficients depend on the beam energy and
/// the object, so each set is only valid for data taken under the conditions it was fitted
/// for; fit your own for anything else.
std::vector<double> get_coefficients(std::string const &phantom_type);

/// Element sigma_t^2 of Sigma between depths a and b (weight (b - u)^2).
double compute_scattering_matrix_element_t(float const &a, float const &b,
                                           std::vector<double> const &coefficients);
/// Element sigma_theta^2 of Sigma between depths a and b (weight 1).
double compute_scattering_matrix_element_theta(float const &a, float const &b,
                                               std::vector<double> const &coefficients);
/// Element sigma_{t theta} of Sigma between depths a and b (weight (b - u)).
double compute_scattering_matrix_element_cross_term(float const &a, float const &b,
                                                    std::vector<double> const &coefficients);

/// Sigma^{-1} between depths a and b. A zero determinant is replaced by a tiny value instead
/// of throwing.
Eigen::Matrix2d construct_inverse_sigma_matrix(float const &a, float const &b,
                                               std::vector<double> const &coefficients);

/// Propagator R = [[1, a - b], [0, 1]] that transports a state (t, theta) over the depth
/// difference a - b in the absence of scattering.
Eigen::Matrix2d construct_rotation_matrix(float const &a, float const &b);

/// Most likely transverse position at each depth in `z`, given the position h and direction
/// theta measured at the entry (z.front()) and exit (z.back()) detectors, for six polynomial
/// coefficients (see get_coefficients() for the built-in sets). The first and last entries of
/// the result equal h_in and h_out; the intermediate depths are evaluated in parallel.
Eigen::VectorXf compute_mlp(std::vector<float> const &z, float const &h_in, float const &theta_in,
                            float const &h_out, float const &theta_out,
                            std::vector<double> const &coefficients, int n_threads = 1);

} // namespace mlp

#endif /* MLP_H */
