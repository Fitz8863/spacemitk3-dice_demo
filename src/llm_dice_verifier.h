#pragma once

#include <string>

enum class LlmWinner {
    Unknown,
    Left,
    Right,
    Tie,
};

// A timeout is distinct from other failures because the caller may use the
// already stable YOLO result as its explicit fallback verdict.
enum class LlmVerificationResult {
    Success,
    Timeout,
    Failure,
};

struct LlmDiceConfig {
    std::string base_url;
    std::string api_key;
    std::string model;
    int timeout_seconds = 20;
    std::string system_prompt;
    std::string user_prompt_template;
};

class LlmDiceVerifier {
public:
    explicit LlmDiceVerifier(LlmDiceConfig config);

    bool configured() const;
    LlmVerificationResult verify_once(const std::string& left_name,
                                      const std::string& right_name,
                                      int left_sum, int right_sum,
                                      LlmWinner& winner,
                                      std::string& error) const;

private:
    LlmDiceConfig config_;
};
