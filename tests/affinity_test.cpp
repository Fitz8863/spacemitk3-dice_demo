#include "affinity.h"
#include <cassert>
#include <iostream>

int main() {
    std::string error;
    assert(validate_ep_affinity("12;13", 2, error));
    assert(validate_ep_affinity("12", 1, error));
    assert(validate_ep_affinity("", 2, error));
    assert(!validate_ep_affinity("12", 2, error));
    assert(!validate_ep_affinity("12;;13", 2, error));
    assert(!validate_ep_affinity("12;cpu", 2, error));
    std::cout << "affinity tests passed\n";
}
