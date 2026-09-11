// Copyright (c) 2026 lucabRTrender contributors.
//
// When a frame happens. SMPTE ST 2059-1 through genlock: frame N of a rate
// begins at an instant anyone who knows the time can compute, so render nodes
// that follow the same PTP master draw frame N for the same instant without
// ever talking to each other.
//
//   free run   this machine's clock is the time
//   PTP        a genlock PtpClock slave follows a master (`genlock-cli master`)
//
// Everything is TAI nanoseconds from the SMPTE epoch, which is what the
// alignment maths runs on; UTC appears only where a person reads a timecode.
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <genlock/Alignment.h>
#include <genlock/Ltc.h>
#include <genlock/PtpClock.h>

#include "lrt/core/Result.h"

namespace lrt::sched {

using Rate = genlock::Rate;
using Timecode = genlock::Timecode;

struct ClockSettings {
    Rate        rate = Rate::k25();
    std::string ptpMaster;   ///< host to follow; empty runs free on this machine's clock
    uint16_t    port = genlock::PtpClock::kDefaultPort;
    uint8_t     domain = 0;
    int64_t     taiMinusUtcSeconds = genlock::Alignment::kDefaultTaiMinusUtcSeconds;
};

/// A frame the clock waited for.
struct Tick {
    int64_t index = 0;        ///< epoch-anchored frame count
    int64_t startTaiNs = 0;   ///< its alignment point
    int64_t wokeTaiNs = 0;    ///< when the wait returned
    /// How late the wait returned: the phase error the render starts with.
    [[nodiscard]] int64_t lateNs() const noexcept { return wokeTaiNs - startTaiNs; }
};

class FrameClock {
public:
    [[nodiscard]] static Result<std::unique_ptr<FrameClock>> create(const ClockSettings& settings);
    ~FrameClock();

    [[nodiscard]] int64_t nowTaiNs() const;
    [[nodiscard]] bool following() const noexcept { return ptp_ != nullptr; }
    /// The slave's lock; all zero when running free.
    [[nodiscard]] genlock::PtpClock::Lock lock() const;
    /// Waits until a PTP slave has locked. Running free, returns at once.
    [[nodiscard]] Result<void> waitForLock(std::chrono::milliseconds timeout) const;

    [[nodiscard]] int64_t frameAt(int64_t taiNs) const noexcept { return alignment_.frameIndexAtTai(taiNs); }
    [[nodiscard]] int64_t frameStartTaiNs(int64_t index) const noexcept { return alignment_.frameStartTai(index); }
    /// Sleeps until frame `index` begins (platform::sleepPrecisely, then the
    /// last 100 µs yielding). A frame already begun returns at once, late.
    [[nodiscard]] Tick waitFor(int64_t index) const;

    /// The time-of-day timecode, UTC, of frame `index`: drop-frame at
    /// 30000/1001 and 60000/1001, as SMPTE 12M counts them from midnight.
    [[nodiscard]] Timecode timecodeOf(int64_t index) const noexcept;
    /// The first frame of the UTC day holding frame `index`: where its
    /// timecodes count from.
    [[nodiscard]] int64_t midnightFrame(int64_t index) const noexcept;
    /// The frame labelled `timecode` nearest frame `near` -- today's, the day
    /// before's or the day after's.
    [[nodiscard]] int64_t frameLabelled(const Timecode& timecode, int64_t near) const noexcept;
    [[nodiscard]] Rate rate() const noexcept { return alignment_.rate(); }
    [[nodiscard]] int64_t taiMinusUtcNs() const noexcept { return taiMinusUtcNs_; }

private:
    FrameClock(const ClockSettings& settings);
    genlock::Alignment                 alignment_;
    int64_t                            taiMinusUtcNs_ = 0;
    std::unique_ptr<genlock::PtpClock> ptp_;
};

/// A timecode from a count of frames since midnight at `rate`: drop-frame at
/// the 1001 rates with a drop-frame count (29.97, 59.94), non-drop otherwise.
[[nodiscard]] Timecode timecodeFromFrames(int64_t framesSinceMidnight, Rate rate) noexcept;

/// The count of frames since midnight a timecode labels: timecodeFromFrames
/// turned round.
[[nodiscard]] int64_t framesFromTimecode(const Timecode& timecode, Rate rate) noexcept;

/// "HH:MM:SS:FF", or with ';' before the frames for drop-frame.
[[nodiscard]] bool parseTimecode(const std::string& text, Timecode& timecode) noexcept;

/// SMPTE 12M's 32-bit BCD packing (OpenEXR's TimeCode, "TV60" flags).
[[nodiscard]] uint32_t packTimecode(const Timecode& timecode) noexcept;

}   // namespace lrt::sched
