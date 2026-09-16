#include "latest_queue.h"
#include <cassert>
#include <iostream>
#include <memory>

int main() {
    LatestQueue<int> queue;
    assert(queue.try_pop_latest() == nullptr);
    assert(!queue.push(std::make_shared<int>(1)));
    assert(queue.push(std::make_shared<int>(2)));
    auto value = queue.try_pop_latest();
    assert(value && *value == 2);
    assert(queue.try_pop_latest() == nullptr);
    queue.push(std::make_shared<int>(3));
    queue.close();
    value = queue.try_pop_latest();
    assert(value && *value == 3);
    assert(queue.try_pop_latest() == nullptr);
    assert(!queue.push(std::make_shared<int>(4)));
    std::cout << "latest queue tests passed\n";
}
