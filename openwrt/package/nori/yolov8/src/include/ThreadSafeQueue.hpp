#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <chrono>

template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue;
    mutable std::mutex mutex;
    std::condition_variable condVar;
    std::condition_variable fullCondVar;
    size_t maxSize;

    // 添加元素到队列
    bool push(const T& value, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        
        if (timeout == std::chrono::milliseconds::min()) {
            // 不等待
            if (queue.size() >= maxSize) {
                return false;
            }
        }
        else if (timeout == std::chrono::milliseconds::max()) {
            // 无限等待直到有空间
            fullCondVar.wait(lock, [this] { return queue.size() < maxSize; });
        } else {
            // 等待直到有空间或超时
            if (!fullCondVar.wait_for(lock, timeout, [this] { return queue.size() < maxSize; })) {
                return false; // 在超时后放弃
            }
        }

        queue.push(value);
        condVar.notify_one();
        return true;
    }

public:
    explicit ThreadSafeQueue(size_t maxSize) : maxSize(maxSize) {}
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;            // 禁止复制构造
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete; // 禁止赋值操作

    bool push(const T& value, int timeoutMs) {
        return push(value, std::chrono::milliseconds(timeoutMs));
    }

    bool push(const T& value) {
        return push(value, std::chrono::milliseconds::max());
    }

    bool immediatelyPush(const T& value) {
        return push(value, std::chrono::milliseconds::min());
    }

    // 从队列中弹出元素，如果队列为空则等待
    T pop() {
        std::unique_lock<std::mutex> lock(mutex);
        condVar.wait(lock, [this]{ return !queue.empty(); });
        T value = std::move(queue.front());
        queue.pop();
        fullCondVar.notify_one();
        return value;
    }

    // 检查队列是否为空
    bool isEmpty() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }

    // 获取队列的大小
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size();
    }
};