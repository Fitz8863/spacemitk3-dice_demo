#include <sherpa-onnx/c-api/c-api.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

static void usage(const char *prog) {
  std::cerr << "Usage: " << prog << " <model_dir> <output.wav> [text] [speed]\n";
}

int main(int argc, char **argv) {
  if (argc < 3) { usage(argv[0]); return 2; }
  const std::string model_dir = argv[1];
  const std::string output = argv[2];
  const std::string text = argc >= 4 ? argv[3] : "你好，这是 C++ 在 SpacemiT 开发板上的语音合成测试。";
  const float speed = argc >= 5 ? std::strtof(argv[4], nullptr) : 1.0f;

  const std::string acoustic = model_dir + "/model-steps-3.q.onnx";
  const std::string vocoder = model_dir + "/vocos-16khz-univ.q.onnx";
  const std::string tokens = model_dir + "/tokens.txt";
  const std::string lexicon = model_dir + "/lexicon.txt";
  const std::string data_dir = model_dir + "/espeak-ng-data";
  const std::string rule_fsts = model_dir + "/date-zh.fst," + model_dir + "/number-zh.fst";

  SherpaOnnxOfflineTtsConfig config{};
  config.model.num_threads = 4;
  config.model.debug = 0;
  config.model.provider = "spacemit";
  config.model.matcha.acoustic_model = acoustic.c_str();
  config.model.matcha.vocoder = vocoder.c_str();
  config.model.matcha.tokens = tokens.c_str();
  config.model.matcha.lexicon = lexicon.c_str();
  config.model.matcha.data_dir = data_dir.c_str();
  config.model.matcha.noise_scale = 0.667f;
  config.model.matcha.length_scale = 1.0f;
  config.rule_fsts = rule_fsts.c_str();
  config.max_num_sentences = 1;
  config.silence_scale = 0.2f;

  std::cerr << "[C++] creating Matcha TTS...\n";
  const auto init_begin = std::chrono::steady_clock::now();
  const SherpaOnnxOfflineTts *tts = SherpaOnnxCreateOfflineTts(&config);
  const auto init_end = std::chrono::steady_clock::now();
  if (!tts) {
    std::cerr << "[C++] failed to create TTS\n";
    return 1;
  }

  SherpaOnnxGenerationConfig generation{};
  generation.sid = 0;
  generation.speed = speed;
  generation.silence_scale = 0.2f;

  std::cerr << "[C++] synthesizing: " << text << "\n";
  const auto synth_begin = std::chrono::steady_clock::now();
  const SherpaOnnxGeneratedAudio *audio =
      SherpaOnnxOfflineTtsGenerateWithConfig(tts, text.c_str(), &generation, nullptr, nullptr);
  const auto synth_end = std::chrono::steady_clock::now();
  if (!audio || !audio->samples || audio->n <= 0) {
    std::cerr << "[C++] synthesis failed\n";
    if (audio) SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
    SherpaOnnxDestroyOfflineTts(tts);
    return 1;
  }

  if (!SherpaOnnxWriteWave(audio->samples, audio->n, audio->sample_rate, output.c_str())) {
    std::cerr << "[C++] failed to write WAV: " << output << "\n";
    SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
    SherpaOnnxDestroyOfflineTts(tts);
    return 1;
  }
  const double init_sec = std::chrono::duration<double>(init_end - init_begin).count();
  const double synth_sec = std::chrono::duration<double>(synth_end - synth_begin).count();
  const double audio_sec = static_cast<double>(audio->n) / audio->sample_rate;
  std::cout << "generated " << output << " samples=" << audio->n
            << " sample_rate=" << audio->sample_rate
            << " duration=" << audio_sec << "s"
            << " init=" << init_sec << "s"
            << " synth=" << synth_sec << "s"
            << " rtf=" << (synth_sec / audio_sec) << "\n";

  SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
  SherpaOnnxDestroyOfflineTts(tts);
  return 0;
}
