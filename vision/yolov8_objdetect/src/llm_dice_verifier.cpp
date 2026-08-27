#include "llm_dice_verifier.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstring>
#include <poll.h>
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

bool read_with_timeout(int fd, int timeout_seconds, std::string& result) {
    result.clear();
    char buffer[4096];
    // curl has its own --max-time, but keep an independent parent-side
    // deadline so a stalled child or inherited pipe can never block the
    // verifier thread (and therefore shutdown) forever.
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(std::max(1, timeout_seconds)) +
                          std::chrono::milliseconds(1000);
    for (;;) {
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        if (remaining.count() <= 0) return false;
        struct pollfd descriptor{fd, POLLIN | POLLHUP | POLLERR, 0};
        const int poll_result = ::poll(&descriptor, 1, static_cast<int>(remaining.count()));
        if (poll_result < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (poll_result == 0) return false;
        const ssize_t n = ::read(fd, buffer, sizeof(buffer));
        if (n > 0) {
            result.append(buffer, static_cast<std::size_t>(n));
        } else if (n == 0) {
            return true;
        } else if (errno != EINTR) {
            return false;
        }
    }
}

void kill_and_reap(pid_t pid, int& status) {
    // The curl process normally has no children, but killing its process group
    // also handles a helper that inherited the output pipe and kept EOF away.
    if (::kill(-pid, SIGKILL) < 0) {
        (void)::kill(pid, SIGKILL);
    }
    pid_t result;
    do {
        result = ::waitpid(pid, &status, 0);
    } while (result < 0 && errno == EINTR);
}

enum class CurlRequestResult {
    Success,
    Timeout,
    Failure,
};

CurlRequestResult run_curl(const std::string& url, const std::string& api_key,
                           int timeout_seconds, const std::string& request,
                           std::string& response, std::string& error) {
    if (api_key.find_first_of("\r\n") != std::string::npos) {
        error = "LLM API key contains an invalid newline";
        return CurlRequestResult::Failure;
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
        return CurlRequestResult::Failure;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        error = std::string("fork failed: ") + std::strerror(errno);
        close_pipe(input_pipe);
        close_pipe(config_pipe);
        close_pipe(output_pipe);
        return CurlRequestResult::Failure;
    }
    if (pid == 0) {
        (void)::setpgid(0, 0);
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

        const std::string connect_timeout = std::to_string(std::min(5, timeout_seconds));
        const std::string request_timeout = std::to_string(timeout_seconds);
        ::execlp("curl", "curl",
                 "--silent", "--show-error", "--fail-with-body",
                 "--connect-timeout", connect_timeout.c_str(),
                 "--max-time", request_timeout.c_str(),
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

    (void)::setpgid(pid, pid);
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

    const bool output_complete = read_with_timeout(output_pipe[0], timeout_seconds, response);
    close_fd(output_pipe[0]);

    int status = 0;
    pid_t wait_result = -1;
    if (!output_complete) {
        // Curl normally enforces --max-time, but do not let a broken/stalled
        // child or inherited pipe keep the verifier thread blocked forever.
        kill_and_reap(pid, status);
        error = "LLM request timed out after " + std::to_string(timeout_seconds) +
                " seconds";
        return CurlRequestResult::Timeout;
    }

    // EOF means curl closed its output descriptors, but allow a short reap
    // window before treating a pathological child as another timeout.
    for (int attempt = 0; attempt < 20; ++attempt) {
        do {
            wait_result = ::waitpid(pid, &status, WNOHANG);
        } while (wait_result < 0 && errno == EINTR);
        if (wait_result != 0) break;
        ::usleep(10000);
    }
    if (wait_result == 0) {
        kill_and_reap(pid, status);
        error = "LLM request timed out after " + std::to_string(timeout_seconds) +
                " seconds";
        return CurlRequestResult::Timeout;
    }

    if (!config_ok || !request_ok) {
        error = "failed to send the LLM request to curl";
        return CurlRequestResult::Failure;
    }
    if (wait_result < 0) {
        error = std::string("waitpid failed: ") + std::strerror(errno);
        return CurlRequestResult::Failure;
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 28) {
        error = "LLM request timed out after " + std::to_string(timeout_seconds) +
                " seconds";
        return CurlRequestResult::Timeout;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        error = response.empty() ? "curl request failed" : response;
        return CurlRequestResult::Failure;
    }
    return CurlRequestResult::Success;
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
           !config_.model.empty() && config_.timeout_seconds >= 1 &&
           !config_.system_prompt.empty() && !config_.user_prompt_template.empty();
}

LlmVerificationResult LlmDiceVerifier::verify_once(
    const std::string& left_name, const std::string& right_name,
    int left_sum, int right_sum, LlmWinner& winner, std::string& error) const {
    winner = LlmWinner::Unknown;
    if (!configured()) {
        error = "LLM is not configured; set llm.api_key in config.json or DICE_LLM_API_KEY";
        return LlmVerificationResult::Failure;
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
    const CurlRequestResult curl_result =
        run_curl(append_chat_completions(config_.base_url), config_.api_key,
                 config_.timeout_seconds, request, response, error);
    if (curl_result == CurlRequestResult::Timeout) return LlmVerificationResult::Timeout;
    if (curl_result != CurlRequestResult::Success) return LlmVerificationResult::Failure;

    winner = parse_winner(response);
    if (winner == LlmWinner::Unknown) {
        error = "LLM response did not contain winner=LEFT, RIGHT, or TIE";
        return LlmVerificationResult::Failure;
    }
    return LlmVerificationResult::Success;
}
