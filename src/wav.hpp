#ifndef LIVEVOICE_WAV_HPP
#define LIVEVOICE_WAV_HPP

// WAV file I/O (dr_wav).

#include <string>
#include <vector>

namespace lv {

// Any PCM / float WAV, down-mixed to mono float.
bool read_wav_mono(const std::string& path, std::vector<float>* pcm, int* sample_rate);

// Mono 16-bit PCM; samples are clamped to [-1, 1].
bool write_wav(const std::string& path, const float* pcm, size_t n, int sample_rate);

}  // namespace lv

#endif  // LIVEVOICE_WAV_HPP
