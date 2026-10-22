#ifndef UTILS_H
#define UTILS_H

#include <string>

#include "blosc2.h"
#include "io.h"

namespace utils {

/// Read a compressed matrix and store it again with a different codec or compression level.
void recompress_matrix(std::string const &filename, std::string const &output, int compression_level = 9,
                       int shuffle_method = BLOSC_SHUFFLE, std::string const &codec = io::DEFAULT_CODEC,
                       int verbosity_level = 0);

/// Randomly keep a fraction `ratio` of the rows of `<input_path>/matrix-<pixel_w_str>/<angle>.mat`
/// and the matching entries of `<input_path>/b/b_<angle>.vec`, writing both under `output_path`.
void subsample_matrix_vector_pair(std::string const &input_path, std::string const &projection_angle,
                                  std::string const &output_path, float ratio, std::string const &pixel_w_str,
                                  unsigned seed = 0, int compression_level = 9,
                                  int shuffle_method = BLOSC_SHUFFLE,
                                  std::string const &codec = io::DEFAULT_CODEC, int verbosity_level = 0);

} // namespace utils
#endif
