#pragma once

#include <string>

bool validate_ep_affinity(const std::string& affinity, int intra_threads,
                          std::string& error);
