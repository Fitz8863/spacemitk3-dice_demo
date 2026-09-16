#pragma once

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

// Single-slot latest-only handoff. Producers replace old work instead of
// waiting behind it, which keeps capture, inference and display latency low.
template <typename T>
class LatestQueue {
public:
    bool push(std::shared_ptr<T> value) {
        if (!value) return false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) return false;
            const bool replaced = static_cast<bool>(pending_);
            pending_ = std::move(value);
            cv_.notify_one();
            return replaced;
        }
    }

    std::shared_ptr<T> try_pop_latest() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto value = std::move(pending_);
        pending_.reset();
        return value;
    }

    std::shared_ptr<T> wait_pop_latest(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, timeout, [this] { return closed_ || pending_; });
        auto value = std::move(pending_);
        pending_.reset();
        return value;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.reset();
    }

    bool closed_and_empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_ && !pending_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::shared_ptr<T> pending_;
    bool closed_ = false;
};
