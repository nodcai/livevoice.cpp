#include "wav.hpp"

#include "common.hpp"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace lv {

bool read_wav_mono(const std::string& path, std::vector<float>* pcm, int* sample_rate) {
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        LV_LOG_ERROR("wav: cannot open %s", path.c_str());
        return false;
    }
    const size_t frames = static_cast<size_t>(wav.totalPCMFrameCount);
    const int    ch     = static_cast<int>(wav.channels);
    std::vector<float> inter(frames * ch);
    const size_t got = drwav_read_pcm_frames_f32(&wav, frames, inter.data());
    if (sample_rate) *sample_rate = static_cast<int>(wav.sampleRate);
    drwav_uninit(&wav);
    pcm->assign(got, 0.0f);
    for (size_t i = 0; i < got; ++i) {
        float s = 0.0f;
        for (int c = 0; c < ch; ++c) s += inter[i * ch + c];
        (*pcm)[i] = s / ch;
    }
    return true;
}

bool write_wav(const std::string& path, const float* pcm, size_t n, int sample_rate) {
    drwav_data_format fmt{};
    fmt.container     = drwav_container_riff;
    fmt.format        = DR_WAVE_FORMAT_PCM;
    fmt.channels      = 1;
    fmt.sampleRate    = static_cast<drwav_uint32>(sample_rate);
    fmt.bitsPerSample = 16;
    drwav wav;
    if (!drwav_init_file_write(&wav, path.c_str(), &fmt, nullptr)) {
        LV_LOG_ERROR("wav: cannot write %s", path.c_str());
        return false;
    }
    std::vector<int16_t> s16(n);
    for (size_t i = 0; i < n; ++i)
        s16[i] = static_cast<int16_t>(std::lround(std::max(-1.0f, std::min(1.0f, pcm[i])) * 32767.0f));
    const bool ok = drwav_write_pcm_frames(&wav, n, s16.data()) == n;
    drwav_uninit(&wav);
    return ok;
}

}  // namespace lv
