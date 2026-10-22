#ifndef PARALLEL_H
#define PARALLEL_H

#include <atomic>
#include <exception>

/// Collects the first exception thrown inside an OpenMP parallel region so it can be rethrown
/// on the calling thread once the region has ended. An exception escaping a parallel region is
/// undefined behaviour and in practice calls std::terminate, which kills the Python interpreter.
///
/// Usage:
///     ExceptionCollector errors;
///     #pragma omp parallel for
///     for (...) errors.run([&] { ...body that may throw... });
///     errors.rethrow();
class ExceptionCollector {
  public:
    template <class F> void run(F &&body) noexcept {
        if (m_failed.load(std::memory_order_relaxed))
            return;
        try {
            body();
        } catch (...) {
#pragma omp critical(pct_tools_exception_collector)
            {
                if (!m_failed.load(std::memory_order_relaxed)) {
                    m_exception = std::current_exception();
                    m_failed.store(true, std::memory_order_relaxed);
                }
            }
        }
    }

    bool failed() const { return m_failed.load(std::memory_order_relaxed); }

    void rethrow() const {
        if (m_failed.load(std::memory_order_relaxed))
            std::rethrow_exception(m_exception);
    }

  private:
    std::atomic<bool> m_failed{false};
    std::exception_ptr m_exception;
};

#endif /* PARALLEL_H */
