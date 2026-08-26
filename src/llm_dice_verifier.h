#pragma once

#include <string>

enum class LlmWinner {
    Unknown,
    Left,
    Right,
    Tie,
};

struct LlmDiceConfig {
    std::string base_url;
    std::string api_key;
    std::string model;
};

class LlmDiceVerifier {
public:
    explicit LlmDiceVerifier(LlmDiceConfig config);

    bool configured() const;
    bool verify_once(const std::string& left_name, const std::string& right_name,
                    int left_sum, int right_sum, LlmWinner& winner,
                    std::string& error) const;

private:
    LlmDiceConfig config_;
};
