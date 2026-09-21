#pragma once

#include <poll.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <string>
#include <vector>

namespace vision_control {

// Read newline-delimited control commands as one batch. A pipe read may
// coalesce adjacent JSONL writes, so callers must drain every complete line
// before polling the kernel fd again.
class CommandReader {
public:
    std::vector<std::string> read_ready(int fd, int timeout_ms) {
        std::vector<std::string> commands;
        drain_complete_lines(commands);
        if (!commands.empty() || fd < 0 || closed_) return commands;

        struct pollfd pfd{fd, POLLIN | POLLHUP | POLLERR, 0};
        int ready = 0;
        do {
            ready = ::poll(&pfd, 1, timeout_ms);
        } while (ready < 0 && errno == EINTR);
        if (ready <= 0) return commands;

        if (pfd.revents & (POLLIN | POLLHUP)) {
            char chunk[4096];
            ssize_t count = 0;
            do {
                count = ::read(fd, chunk, sizeof(chunk));
            } while (count < 0 && errno == EINTR);
            if (count > 0) {
                buffer_.append(chunk, static_cast<std::size_t>(count));
                drain_complete_lines(commands);
            } else if (count == 0 || (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
                closed_ = true;
            }
        } else if (pfd.revents & (POLLERR | POLLNVAL)) {
            closed_ = true;
        }
        return commands;
    }

    bool closed() const { return closed_; }

private:
    void drain_complete_lines(std::vector<std::string>& commands) {
        std::size_t newline = std::string::npos;
        while ((newline = buffer_.find('\n')) != std::string::npos) {
            commands.push_back(buffer_.substr(0, newline));
            buffer_.erase(0, newline + 1);
        }
    }

    std::string buffer_;
    bool closed_ = false;
};

}  // namespace vision_control
