#include "llm_dice_verifier.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <csignal>
#include <cstring>
#include <regex>
#include <sstream>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

namespace {

std::string json_escape(const std::string& value) {
    std::string result;
    result.reserve(value.size() + 8);
    for (const unsigned char c : value) {
        switch (c) {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (c < 0x20) {
                constexpr char hex[] = "0123456789abcdef";
                result += "\\u00";
                result.push_back(hex[(c >> 4) & 0x0f]);
                result.push_back(hex[c & 0x0f]);
            } else {
                result.push_back(static_cast<char>(c));
            }
        }
    }
    return result;
}


void replace_all(std::string& text, const std::string& token,
                 const std::string& value) {
    std::size_t position = 0;
    while ((position = text.find(token, position)) != std::string::npos) {
        text.replace(position, token.size(), value);
        position += value.size();
    }
}

std::string curl_config_escape(const std::string& value) {
    std::string result;
    result.reserve(value.size() + 8);
    for (const char c : value) {
        if (c == '\\' || c == '"') result.push_back('\\');
        result.push_back(c);
    }
    return result;
}

void close_fd(int& fd) {
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

void close_pipe(int pipe_fds[2]) {
    close_fd(pipe_fds[0]);
    close_fd(pipe_fds[1]);
}

bool write_all(int fd, const std::string& data) {
    std::size_t offset = 0;
    while (offset < data.size()) {
        const ssize_t written = ::write(fd, data.data() + offset, data.size() - offset);
        if (written < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        offset += static_cast<std::size_t>(written);
    }
    return true;
}

std::string read_all(int fd) {
    std::string result;
    char buffer[4096];
    for (;;) {
        const ssize_t n = ::read(fd, buffer, sizeof(buffer));
        if (n > 0) {
            result.append(buffer, static_cast<std::size_t>(n));
        } else if (n == 0) {
            break;
        } else if (errno != EINTR) {
            break;
        }
    }
    return result;
}

bool run_curl(const std::string& url, const std::string& api_key,
              const std::string& request, std::string& response,
              std::string& error) {
    if (api_key.find_first_of("\r\n") != std::string::npos) {
        error = "DICE_LLM_API_KEY contains an invalid newline";
        return false;
    }

    std::signal(SIGPIPE, SIG_IGN);
    int input_pipe[2] = {-1, -1};
    int config_pipe[2] = {-1, -1};
    int output_pipe[2] = {-1, -1};
    if (::pipe(input_pipe) != 0 || ::pipe(config_pipe) != 0 || ::pipe(output_pipe) != 0) {
        error = std::string("pipe failed: ") + std::strerror(errno);
        close_pipe(input_pipe);
        close_pipe(config_pipe);
        close_pipe(output_pipe);
        return false;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        error = std::string("fork failed: ") + std::strerror(errno);
        close_pipe(input_pipe);
        close_pipe(config_pipe);
        close_pipe(output_pipe);
        return false;
    }
    if (pid == 0) {
        constexpr int kCurlConfigFd = 10;
        if (::dup2(input_pipe[0], STDIN_FILENO) < 0 ||
            ::dup2(output_pipe[1], STDOUT_FILENO) < 0 ||
            ::dup2(output_pipe[1], STDERR_FILENO) < 0 ||
            ::dup2(config_pipe[0], kCurlConfigFd) < 0) {
            _exit(126);
        }
        close_pipe(input_pipe);
        close_pipe(config_pipe);
        close_pipe(output_pipe);

        ::execlp("curl", "curl",
                 "--silent", "--show-error", "--fail-with-body",
                 "--connect-timeout", "5", "--max-time", "20",
                 "--request", "POST",
                 "--config", "/proc/self/fd/10",
                 "--header", "Content-Type: application/json",
                 "--header", "Accept: application/json",
                 "--data-binary", "@-", url.c_str(),
                 static_cast<char*>(nullptr));
        const std::string message =
            std::string("exec curl failed: ") + std::strerror(errno);
        const ssize_t ignored = ::write(STDERR_FILENO, message.data(), message.size());
        (void)ignored;
        _exit(127);
    }

    close_fd(input_pipe[0]);
    close_fd(config_pipe[0]);
    close_fd(output_pipe[1]);

    // Pass the secret header through an inherited pipe instead of curl argv,
    // so the API key is not exposed through /proc/<pid>/cmdline or process tools.
    const std::string curl_config =
        "header = \"Authorization: Bearer " + curl_config_escape(api_key) + "\"\n";
    const bool config_ok = write_all(config_pipe[1], curl_config);
    close_fd(config_pipe[1]);
    const bool request_ok = write_all(input_pipe[1], request);
    close_fd(input_pipe[1]);

    response = read_all(output_pipe[0]);
    close_fd(output_pipe[0]);

    int status = 0;
    pid_t wait_result;
    do {
        wait_result = ::waitpid(pid, &status, 0);
    } while (wait_result < 0 && errno == EINTR);

    if (!config_ok || !request_ok) {
        error = "failed to send the LLM request to curl";
        return false;
    }
    if (wait_result < 0) {
        error = std::string("waitpid failed: ") + std::strerror(errno);
        return false;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        error = response.empty() ? "curl request failed" : response;
        return false;
    }
    return true;
}

std::string append_chat_completions(std::string base_url) {
    while (!base_url.empty() && base_url.back() == '/') base_url.pop_back();
    constexpr const char* suffix = "/chat/completions";
    constexpr std::size_t suffix_length = 17;
    if (base_url.size() >= suffix_length &&
        base_url.compare(base_url.size() - suffix_length, suffix_length, suffix) == 0) {
        return base_url;
    }
    return base_url + suffix;
}

std::string extract_json_string(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) return {};
    std::size_t pos = json.find(':', key_pos + needle.size());
    if (pos == std::string::npos) return {};
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return {};
    ++pos;
    std::string value;
    bool escaped = false;
    for (; pos < json.size(); ++pos) {
        const char c = json[pos];
        if (escaped) {
            switch (c) {
            case 'n': value.push_back('\n'); break;
            case 'r': value.push_back('\r'); break;
            case 't': value.push_back('\t'); break;
            default: value.push_back(c); break;
            }
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (c == '"') {
            return value;
        } else {
            value.push_back(c);
        }
    }
    return {};
}

LlmWinner winner_from_token(std::string token) {
    std::transform(token.begin(), token.end(), token.begin(),
                   [](const unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (token == "LEFT") return LlmWinner::Left;
    if (token == "RIGHT") return LlmWinner::Right;
    if (token == "TIE") return LlmWinner::Tie;
    return LlmWinner::Unknown;
}

LlmWinner parse_winner(const std::string& response) {
    const std::string content = extract_json_string(response, "content");
    const std::regex winner_regex(R"REGEX("winner"\s*:\s*"(LEFT|RIGHT|TIE)")REGEX",
                                  std::regex_constants::icase);
    std::smatch match;
    if (std::regex_search(content, match, winner_regex) ||
        std::regex_search(response, match, winner_regex)) {
        return winner_from_token(match[1].str());
    }

    const std::regex token_regex(R"(\b(LEFT|RIGHT|TIE)\b)",
                                 std::regex_constants::icase);
    if (std::regex_search(content, match, token_regex) ||
        std::regex_search(response, match, token_regex)) {
        return winner_from_token(match[1].str());
    }
    return LlmWinner::Unknown;
}

}  // namespace

LlmDiceVerifier::LlmDiceVerifier(LlmDiceConfig config) : config_(std::move(config)) {}

bool LlmDiceVerifier::configured() const {
    return !config_.base_url.empty() && !config_.api_key.empty() &&
           !config_.model.empty() && !config_.system_prompt.empty() &&
           !config_.user_prompt_template.empty();
}

bool LlmDiceVerifier::verify_once(const std::string& left_name, const std::string& right_name,
                                  int left_sum, int right_sum, LlmWinner& winner,
                                  std::string& error) const {
    winner = LlmWinner::Unknown;
    if (!configured()) {
        error = "LLM is not configured; set DICE_LLM_API_KEY";
        return false;
    }

    std::string user_prompt = config_.user_prompt_template;
    replace_all(user_prompt, "{left_name}", left_name);
    replace_all(user_prompt, "{right_name}", right_name);
    replace_all(user_prompt, "{left_sum}", std::to_string(left_sum));
    replace_all(user_prompt, "{right_sum}", std::to_string(right_sum));

    const std::string request =
        "{\"model\":\"" + json_escape(config_.model) + "\","
        "\"messages\":["
        "{\"role\":\"system\",\"content\":\"" +
        json_escape(config_.system_prompt) + "\"},"
        "{\"role\":\"user\",\"content\":\"" +
        json_escape(user_prompt) + "\"}]}";

    std::string response;
    if (!run_curl(append_chat_completions(config_.base_url), config_.api_key,
                  request, response, error)) {
        return false;
    }
    winner = parse_winner(response);
    if (winner == LlmWinner::Unknown) {
        error = "LLM response did not contain winner=LEFT, RIGHT, or TIE";
        return false;
    }
    return true;
}
