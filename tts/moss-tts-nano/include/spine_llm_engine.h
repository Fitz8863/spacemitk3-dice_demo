// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT

#include "ggml.h"
#include "llama.h"
#include "spine_llm_argparser.h"
#include <string>
#include <vector>

namespace onnxruntime {
namespace spacemit {
namespace {
enum class SpineArchitecture {
    UNKNOWN,
    LLaVA_Qwen2_ForCausalLM,
    // TODO: QwenVL, etc.
};

SpineArchitecture GetArchitectureFromString(const std::string & arch_name);
}  // namespace

class SpineLLMEngine {
  public:
    SpineLLMEngine(std::string & model_path) { llm_model_path_ = model_path; }

    ~SpineLLMEngine() { llama_sampler_free(llm_smpl_); }

    bool InitSpineLLMEngine(SpineModelConfig config);

    llama_model * GetLLMModel() { return llm_model_ptr_; }

    llama_context * GetLLMContext() { return llm_context_ptr_; }

    void                 SetTokenEmbeddings(SpineModelConfig config);
    std::vector<float> & EncodeUserText(const std::string & text);
    std::vector<float> & MultimodalEmbeddingFusion(const std::vector<float> & image_emb,
                                                   const std::vector<float> & text_emb,
                                                   std::string &              architecture);
    bool        LLMPrefill(llama_context * ctx, std::vector<float> & embeddings, int hidden_size, llama_pos & n_past);
    std::string LLMDecode(llama_context * ctx, llama_model * model, llama_pos n_past);
    std::string Run(std::string & user_text, std::vector<float> & image_embedding, SpineModelConfig config);

    int llm_context_size_       = 4096;
    int llm_logical_batch_size_ = 2048;
    int llm_predict_token_      = -1;

  private:
    std::string        llm_model_path_;
    llama_model *      llm_model_ptr_{ nullptr };
    llama_context *    llm_context_ptr_{ nullptr };
    std::vector<float> token_embeddings_;
    std::vector<float> text_embeddings_;
    std::vector<float> multimodal_embeddings_;
    int64_t            vocab_size_;
    int64_t            hidden_size_;

    // 用于跟踪 <image> 位置的分割信息
    int n_tokens_before_image_ = 0;  // <image> 之前的 token 数量
    int n_tokens_after_image_  = 0;  // <image> 之后的 token 数量

    llama_model_params   llama_params_;
    const llama_vocab *  llama_vocab_;
    llama_context_params ctx_params_;
    llama_sampler *      llm_smpl_{ nullptr };
};

}  // namespace spacemit
}  // namespace onnxruntime
