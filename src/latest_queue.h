// 单槽 latest-only 队列
//
// 设计照搬参考工程 dice-game/yolov8_segdetect/src/latest_queue.h：
// 生产者**替换**旧任务而不是排队等待，所以采集永不阻塞，
// 慢的阶段只是跳过旧帧、永远处理最新帧。
//
// 为什么对实时视觉是对的：宁可丢旧帧，也不要让延迟累积。
// 如果改成有界队列，一旦某级变慢，延迟就会随着队列长度线性增长。
#pragma once

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

namespace yuan {

template <typename T>
class LatestQueue {
public:
    // 返回 true 表示替换掉了尚未消费的旧任务（说明下游没跟上）
    bool push(std::shared_ptr<T> value) {
        if (!value) return false;
        std::lock_guard<std::mutex> lk(mutex_);
        if (closed_) return false;
        const bool replaced = static_cast<bool>(pending_);
        pending_ = std::move(value);
        cv_.notify_one();
        return replaced;
    }

    // 非阻塞取最新
    std::shared_ptr<T> try_pop_latest() {
        std::lock_guard<std::mutex> lk(mutex_);
        auto v = std::move(pending_);
        pending_.reset();
        return v;
    }

    // 阻塞至多 timeout，取最新（可能返回空）
    std::shared_ptr<T> wait_pop_latest(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_.wait_for(lk, timeout, [this] { return closed_ || pending_; });
        auto v = std::move(pending_);
        pending_.reset();
        return v;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    bool closed_and_empty() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return closed_ && !pending_;
    }

private:
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    std::shared_ptr<T>      pending_;
    bool                    closed_ = false;
};

}  // namespace yuan
