#include "affinity.h"

#include <algorithm>
#include <cctype>
#include <sstream>

bool validate_ep_affinity(const std::string& affinity, int intra_threads,
                          std::string& error) {
    if (intra_threads < 1) {
        error = "intra_threads must be >= 1";
        return false;
    }
    if (affinity.empty()) return true;
    if (affinity.front() == ';' || affinity.back() == ';' || affinity.find(";;") != std::string::npos) {
        error = "ep_affinity contains an empty core ID";
        return false;
    }
    std::size_t count = 0;
    std::size_t begin = 0;
    while (begin < affinity.size()) {
        const std::size_t end = affinity.find(';', begin);
        const std::string token = affinity.substr(begin, end == std::string::npos ? end : end - begin);
        if (token.empty() || !std::all_of(token.begin(), token.end(),
                                          [](unsigned char c) { return std::isdigit(c) != 0; })) {
            error = "ep_affinity must be a semicolon-separated list of numeric core IDs";
            return false;
        }
        ++count;
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    if (count != static_cast<std::size_t>(intra_threads)) {
        std::ostringstream message;
        message << "ep_affinity contains " << count << " core IDs, but intra_threads is "
                << intra_threads;
        error = message.str();
        return false;
    }
    return true;
}
