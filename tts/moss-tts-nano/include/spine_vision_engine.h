// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT

#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include <map>
#include <string>

namespace onnxruntime {
namespace spacemit {

struct RunConfig {
    std::basic_string<ORTCHAR_T>                    profile_file;
    bool                                            f_dump_statistics{ false };
    int                                             random_seed_for_input_data{ -1 };
    bool                                            f_verbose{ false };
    bool                                            enable_memory_pattern{ true };
    bool                                            enable_cpu_mem_arena{ true };
    bool                                            generate_model_input_binding{ false };
    ExecutionMode                                   execution_mode{ ExecutionMode::ORT_SEQUENTIAL };
    int                                             intra_op_num_threads{ 4 };
    int                                             inter_op_num_threads{ 0 };
    GraphOptimizationLevel                          optimization_level{ ORT_ENABLE_ALL };
    std::basic_string<ORTCHAR_T>                    optimized_model_path;
    bool                                            set_denormal_as_zero{ false };
    std::basic_string<ORTCHAR_T>                    ep_runtime_config_string;
    std::unordered_map<std::string, std::string>    session_config_entries;
    std::map<std::basic_string<ORTCHAR_T>, int64_t> free_dim_name_overrides;
    std::map<std::basic_string<ORTCHAR_T>, int64_t> free_dim_denotation_overrides;
    std::string                                     intra_op_thread_affinities;
    bool                                            disable_spinning             = false;
    bool                                            disable_spinning_between_run = false;
    bool                                            exit_after_session_creation  = false;
    std::basic_string<ORTCHAR_T>                    register_custom_op_path;
};

class SpineVisionModelEngine {
  public:
    SpineVisionModelEngine(std::string & model_path) { model_file_path_ = model_path; }

    SpineVisionModelEngine(std::string & model_path, int intra_threads, int inter_threads) :
        SpineVisionModelEngine(model_path, intra_threads, inter_threads, "") {}

    SpineVisionModelEngine(std::string &       model_path,
                           int                 intra_threads,
                           int                 inter_threads,
                           const std::string & affinity) :
        model_file_path_(model_path) {
        ep_config_["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(intra_threads);
        ep_config_["SPACEMIT_EP_INTER_THREAD_NUM"] = std::to_string(inter_threads);
        if (!affinity.empty()) {
            ep_config_["SPACEMIT_EP_INTRA_THREAD_AFFINITY"] = affinity;
        }
    }

    SpineVisionModelEngine(std::string &                                        model_path,
                           const std::string &                                  architecture,
                           const std::unordered_map<std::string, std::string> & ep_config) :
        model_file_path_(model_path),
        architecture_(architecture) {
        ep_config_ = ep_config;
    }

    void SetArchitecture(const std::string & architecture) { architecture_ = architecture; }

    ~SpineVisionModelEngine() { ; }

    Ort::Session & CreateVisionModelSession();

    void InitSesstionOptions();

    std::vector<float> RunSession(Ort::Value & input_tensor);

    Ort::Value & SetInputTensor(std::string & input_binary_path);

  private:
    Ort::Env                                     spine_llm_env_{ nullptr };
    Ort::SessionOptions                          session_options_;
    std::string                                  model_file_path_;
    RunConfig                                    run_config_;
    Ort::Session                                 vision_session_{ nullptr };
    std::string                                  architecture_;
    std::unordered_map<std::string, std::string> ep_config_;
    std::vector<std::string>                     output_names_;
    std::vector<const char *>                    output_names_raw_ptr_;
    std::vector<const char *>                    input_names_;
    std::vector<std::string>                     input_names_str_;
    Ort::Value                                   input_tensor_;
    std::vector<float>                           vision_binary_data_;
};

}  // namespace spacemit
}  // namespace onnxruntime
