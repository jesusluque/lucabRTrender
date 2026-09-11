// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <cassert>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

#include "lrt/core/Error.h"

namespace lrt {

/// A value or an Error. `std::expected<T, Error>` in spirit, hand-rolled for
/// the reason openFXplayer gives: the project is C++20 and whether libc++ has
/// `std::expected` is not something the build should depend on.
///
/// Errors are returned, never thrown. Where a library throws (OpenUSD can),
/// the exception is caught at that module's boundary and turned into this.
template <typename T>
class [[nodiscard]] Result {
    static_assert(!std::is_same_v<std::remove_cv_t<T>, Error>,
                  "Result<Error> is meaningless; use Result<void>");

public:
    using value_type = T;

    Result(T value) : storage_(std::move(value)) {}       // NOLINT(*-explicit-*)
    Result(Error error) : storage_(std::move(error)) {}   // NOLINT(*-explicit-*)

    [[nodiscard]] bool hasValue() const noexcept { return storage_.index() == 0; }
    explicit operator bool() const noexcept { return hasValue(); }

    [[nodiscard]] T& value() & {
        assert(hasValue() && "Result::value() on an error");
        return std::get<0>(storage_);
    }
    [[nodiscard]] const T& value() const& {
        assert(hasValue() && "Result::value() on an error");
        return std::get<0>(storage_);
    }
    [[nodiscard]] T&& value() && {
        assert(hasValue() && "Result::value() on an error");
        return std::get<0>(std::move(storage_));
    }
    [[nodiscard]] const Error& error() const& {
        assert(!hasValue() && "Result::error() on a value");
        return std::get<1>(storage_);
    }
    [[nodiscard]] Error&& error() && {
        assert(!hasValue() && "Result::error() on a value");
        return std::get<1>(std::move(storage_));
    }

    T& operator*() & { return value(); }
    const T& operator*() const& { return value(); }
    T* operator->() { return &value(); }
    const T* operator->() const { return &value(); }

private:
    std::variant<T, Error> storage_;
};

template <>
class [[nodiscard]] Result<void> {
public:
    using value_type = void;

    Result() = default;
    Result(Error error) : error_(std::move(error)) {}     // NOLINT(*-explicit-*)

    [[nodiscard]] bool hasValue() const noexcept { return !error_.has_value(); }
    explicit operator bool() const noexcept { return hasValue(); }
    [[nodiscard]] const Error& error() const& {
        assert(!hasValue() && "Result::error() on a value");
        return *error_;
    }
    [[nodiscard]] Error&& error() && {
        assert(!hasValue() && "Result::error() on a value");
        return std::move(*error_);
    }

private:
    std::optional<Error> error_;
};

inline Result<void> ok() { return {}; }

/// Returns the error of `expr` from the enclosing function if it has one.
#define LRT_TRY(expr)                                   \
    do {                                                \
        if (auto lrt_try_ = (expr); !lrt_try_) {        \
            return std::move(lrt_try_).error();         \
        }                                               \
    } while (false)

}   // namespace lrt
