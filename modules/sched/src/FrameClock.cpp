// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/sched/FrameClock.h"

#include <cstdio>
#include <cstdlib>
#include <thread>

#include "lrt/core/Platform.h"

namespace lrt::sched {
namespace {

constexpr int64_t kSecond = 1000000000LL;
constexpr int64_t kDay = 86400;

bool dropFrameRate(Rate rate) {
    return rate.denominator == 1001 && (rate.numerator == 30000 || rate.numerator == 60000);
}

/// Frames a timecode counts per second: the rate rounded up (30 at 29.97).
int64_t nominal(Rate rate) {
    return (rate.numerator + rate.denominator - 1) / rate.denominator;
}

int64_t floorDiv(int64_t a, int64_t b) {
    return a / b - ((a % b != 0) && ((a < 0) != (b < 0)) ? 1 : 0);
}

}   // namespace

FrameClock::FrameClock(const ClockSettings& settings)
    : alignment_(settings.rate, settings.taiMinusUtcSeconds),
      taiMinusUtcNs_(settings.taiMinusUtcSeconds * kSecond) {}

FrameClock::~FrameClock() {
    if (ptp_ != nullptr) {
        ptp_->stop();
    }
}

Result<std::unique_ptr<FrameClock>> FrameClock::create(const ClockSettings& settings) {
    if (!settings.rate.valid()) {
        return Error(ErrorCode::InvalidArgument, "a frame rate needs a positive numerator and denominator");
    }
    std::unique_ptr<FrameClock> clock(new FrameClock(settings));
    if (!settings.ptpMaster.empty()) {
        clock->ptp_ = std::make_unique<genlock::PtpClock>();
        const std::string why = clock->ptp_->startSlave(settings.ptpMaster, settings.port, settings.domain);
        if (!why.empty()) {
            return Error(ErrorCode::IoFailure, "PTP slave of " + settings.ptpMaster + ": " + why);
        }
    }
    return clock;
}

int64_t FrameClock::nowTaiNs() const {
    const uint64_t utc = ptp_ != nullptr ? ptp_->masterNowNs() : genlock::PtpClock::systemNowNs();
    return static_cast<int64_t>(utc) + taiMinusUtcNs_;
}

genlock::PtpClock::Lock FrameClock::lock() const {
    return ptp_ != nullptr ? ptp_->lock() : genlock::PtpClock::Lock{};
}

Result<void> FrameClock::waitForLock(std::chrono::milliseconds timeout) const {
    if (ptp_ == nullptr) {
        return ok();
    }
    const auto until = std::chrono::steady_clock::now() + timeout;
    while (!ptp_->lock().locked()) {
        if (std::chrono::steady_clock::now() >= until) {
            return Error(ErrorCode::Cancelled, "no PTP lock within the timeout");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return ok();
}

Tick FrameClock::waitFor(int64_t index) const {
    Tick tick;
    tick.index = index;
    tick.startTaiNs = frameStartTaiNs(index);
    for (;;) {
        const int64_t left = tick.startTaiNs - nowTaiNs();
        if (left <= 0) {
            break;
        }
        if (left > 200'000) {
            // Short of the point: the clock is read again, since a PTP
            // correction may have moved it meanwhile.
            platform::sleepPrecisely(std::chrono::nanoseconds(left - 100'000));
        } else {
            std::this_thread::yield();
        }
    }
    tick.wokeTaiNs = nowTaiNs();
    return tick;
}

int64_t FrameClock::midnightFrame(int64_t index) const noexcept {
    // The frame's start in UTC, and UTC's midnight before it. Frame
    // boundaries fall on whole seconds only at whole rates: the day's count
    // starts at the first frame beginning at or after midnight.
    const int64_t utc = frameStartTaiNs(index) - taiMinusUtcNs_;
    const int64_t midnightTai = floorDiv(utc, kDay * kSecond) * kDay * kSecond + taiMinusUtcNs_;
    const int64_t holding = alignment_.frameIndexAtTai(midnightTai);
    return frameStartTaiNs(holding) < midnightTai ? holding + 1 : holding;
}

Timecode FrameClock::timecodeOf(int64_t index) const noexcept {
    return timecodeFromFrames(index - midnightFrame(index), rate());
}

int64_t FrameClock::frameLabelled(const Timecode& timecode, int64_t near) const noexcept {
    const int64_t count = framesFromTimecode(timecode, rate());
    const int64_t today = midnightFrame(near);
    int64_t best = today + count;
    for (const int64_t day : {-1, 1}) {
        // The neighbouring day's midnight: a frame a day's worth away, found again.
        const int64_t other = midnightFrame(frameAt(frameStartTaiNs(today) + day * kDay * kSecond)) + count;
        if (std::llabs(other - near) < std::llabs(best - near)) {
            best = other;
        }
    }
    return best;
}

int64_t framesFromTimecode(const Timecode& tc, Rate rate) noexcept {
    const int64_t fps = nominal(rate);
    int64_t frames = ((int64_t{tc.hours} * 60 + tc.minutes) * 60 + tc.seconds) * fps + tc.frames;
    if (dropFrameRate(rate)) {
        const int64_t minutes = int64_t{tc.hours} * 60 + tc.minutes;
        frames -= (fps / 15) * (minutes - minutes / 10);
    }
    return frames;
}

bool parseTimecode(const std::string& text, Timecode& tc) noexcept {
    int h = 0, m = 0, s = 0, f = 0;
    char separator = 0;
    if (std::sscanf(text.c_str(), "%d:%d:%d%c%d", &h, &m, &s, &separator, &f) != 5 ||
        (separator != ':' && separator != ';') || h < 0 || h > 23 || m < 0 || m > 59 || s < 0 || s > 59 || f < 0) {
        return false;
    }
    tc = Timecode{};
    tc.hours = h;
    tc.minutes = m;
    tc.seconds = s;
    tc.frames = f;
    tc.dropFrame = separator == ';';
    return true;
}

Timecode timecodeFromFrames(int64_t frames, Rate rate) noexcept {
    Timecode tc;
    const int64_t fps = nominal(rate);
    if (dropFrameRate(rate)) {
        // SMPTE 12M: labels ;00 and ;01 (;00-;03 at 59.94) are skipped at the
        // start of every minute not a multiple of ten.
        tc.dropFrame = true;
        const int64_t drop = fps / 15;
        const int64_t perTenMinutes = fps * 600 - drop * 9;
        const int64_t perMinute = fps * 60 - drop;
        frames = ((frames % (perTenMinutes * 144)) + perTenMinutes * 144) % (perTenMinutes * 144);
        const int64_t tens = frames / perTenMinutes;
        const int64_t rest = frames % perTenMinutes;
        frames += drop * 9 * tens + (rest > drop ? drop * ((rest - drop) / perMinute) : 0);
    }
    const int64_t perDay = fps * kDay;
    frames = ((frames % perDay) + perDay) % perDay;
    tc.frames = static_cast<int>(frames % fps);
    const int64_t seconds = frames / fps;
    tc.seconds = static_cast<int>(seconds % 60);
    tc.minutes = static_cast<int>((seconds / 60) % 60);
    tc.hours = static_cast<int>(seconds / 3600);
    return tc;
}

uint32_t packTimecode(const Timecode& tc) noexcept {
    const auto bcd = [](int value, int shift, int unitsBits, int tensBits) {
        const auto units = static_cast<uint32_t>(value % 10) & ((1u << unitsBits) - 1);
        const auto tens = static_cast<uint32_t>(value / 10) & ((1u << tensBits) - 1);
        return (units | (tens << unitsBits)) << shift;
    };
    uint32_t packed = 0;
    packed |= bcd(tc.frames, 0, 4, 2);
    packed |= tc.dropFrame ? (1u << 6) : 0;
    packed |= tc.colourFrame ? (1u << 7) : 0;
    packed |= bcd(tc.seconds, 8, 4, 3);
    packed |= bcd(tc.minutes, 16, 4, 3);
    packed |= bcd(tc.hours, 24, 4, 2);
    return packed;
}

}   // namespace lrt::sched
