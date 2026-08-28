// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT

#include <string>
#include <vector>

namespace onnxruntime {
namespace spacemit {
struct SpineModelConfig {
    std::vector<std::string> architectures;
    std::string              vision_model_path;
    std::string              llm_model_path;
    std::string              token_embedding_path;
    int                      context_size       = 4096;
    int                      logical_batch_size = 2048;
    int                      predict_token      = 128;
    float                    temp               = 0.8;
    int                      top_k              = 40;
    float                    top_p              = 0.9;
    float                    repeat_penalty     = 1.1;
    int64_t                  vocab_size         = 151936;
    int64_t                  hidden_size        = 896;
};

class SpineLLMArgParser {
  public:
    static bool LoadConfigFromDir(const std::string & data_dir, SpineModelConfig & config);

  private:
    static std::string ReadFileToString(const std::string & path);
    static std::string ExtractStringValue(const std::string & content, const std::string & key);
    static int         ExtractIntValue(const std::string & content, const std::string & key, int default_val);
    static int64_t ExtractInt64Value(const std::string & json_block, const std::string & key, int64_t default_value);
    static size_t  FindClosingBrace(const std::string & str, size_t startPos);
    static std::vector<std::string> ExtractStringArray(const std::string & content, const std::string & key);
    static std::string              Trim(const std::string & str);
    static std::string              NormalizePath(const std::string & base_dir, const std::string & rel_path);
};

}  // namespace spacemit
}  // namespace onnxruntime
