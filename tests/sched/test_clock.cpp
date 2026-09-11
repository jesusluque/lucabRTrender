// Copyright (c) 2026 lucabRTrender contributors.
//
// The frame clock: timecodes counted as SMPTE 12M counts them, frames waited
// for on this machine's clock, and on a PTP master's over loopback.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <vector>

#include "lrt/io/Exr.h"
#include "lrt/sched/FrameClock.h"

using namespace lrt;

namespace {

constexpr int64_t kSecond = 1000000000LL;
/// 2026-01-01T00:00:00 UTC, seconds since the Unix epoch.
constexpr int64_t kNewYear2026 = 1767225600;

sched::Timecode tc(int h, int m, int s, int f, bool drop = false) {
    sched::Timecode t;
    t.hours = h;
    t.minutes = m;
    t.seconds = s;
    t.frames = f;
    t.dropFrame = drop;
    return t;
}

}   // namespace

TEST_CASE("drop-frame timecode skips the labels SMPTE 12M skips", "[sched]") {
    const sched::Rate ntsc = sched::Rate::k2997();
    CHECK(sched::timecodeFromFrames(0, ntsc) == tc(0, 0, 0, 0, true));
    CHECK(sched::timecodeFromFrames(1799, ntsc) == tc(0, 0, 59, 29, true));
    CHECK(sched::timecodeFromFrames(1800, ntsc) == tc(0, 1, 0, 2, true));
    CHECK(sched::timecodeFromFrames(17981, ntsc) == tc(0, 9, 59, 29, true));
    CHECK(sched::timecodeFromFrames(17982, ntsc) == tc(0, 10, 0, 0, true));
    CHECK(sched::timecodeFromFrames(17982 * 6, ntsc) == tc(1, 0, 0, 0, true));
    const sched::Rate ntsc60 = sched::Rate::k5994();
    CHECK(sched::timecodeFromFrames(3600, ntsc60) == tc(0, 1, 0, 4, true));
    // Whole rates, and 23.976, count without dropping.
    CHECK(sched::timecodeFromFrames(25 * 3600 + 7, sched::Rate::k25()) == tc(1, 0, 0, 7));
    CHECK(sched::timecodeFromFrames(24 * 61, sched::Rate{24000, 1001}) == tc(0, 1, 1, 0));
}

TEST_CASE("a timecode's frame count is the count that labels it", "[sched]") {
    for (const sched::Rate rate : {sched::Rate::k25(), sched::Rate::k2997(), sched::Rate::k5994()}) {
        for (const int64_t frames : {int64_t{0}, int64_t{1799}, int64_t{1800}, int64_t{17982}, int64_t{123456},
                                     int64_t{2000000}}) {
            CHECK(sched::framesFromTimecode(sched::timecodeFromFrames(frames, rate), rate) == frames);
        }
    }
    sched::Timecode parsed;
    REQUIRE(sched::parseTimecode("10:20:30;12", parsed));
    CHECK(parsed == tc(10, 20, 30, 12, true));
    CHECK_FALSE(sched::parseTimecode("25:00:00:00", parsed));

    auto clock = sched::FrameClock::create({});
    REQUIRE(clock);
    // Near 00:00:01 the frame labelled 23:59:59:00 is yesterday's.
    const int64_t justAfter = (*clock)->frameAt(kNewYear2026 * kSecond + (*clock)->taiMinusUtcNs()) + 25;
    const int64_t late = (*clock)->frameLabelled(tc(23, 59, 59, 0), justAfter);
    CHECK(late == justAfter - 50);
    CHECK((*clock)->timecodeOf(late) == tc(23, 59, 59, 0));
}

TEST_CASE("timecodes pack into SMPTE 12M's BCD", "[sched]") {
    CHECK(sched::packTimecode(tc(10, 20, 30, 12)) == 0x10203012u);
    CHECK(sched::packTimecode(tc(23, 59, 59, 29, true)) == (0x23595929u | (1u << 6)));
}

TEST_CASE("a frame's timecode is its UTC time of day", "[sched]") {
    auto clock = sched::FrameClock::create({});
    REQUIRE(clock);
    // 10:20:30 and twelve frames of 25, as TAI.
    const int64_t utc = (kNewYear2026 + 10 * 3600 + 20 * 60 + 30) * kSecond + 12 * (kSecond / 25);
    const int64_t index = (*clock)->frameAt(utc + (*clock)->taiMinusUtcNs() + 1000);
    CHECK((*clock)->timecodeOf(index) == tc(10, 20, 30, 12));
    CHECK((*clock)->frameStartTaiNs(index) - (*clock)->taiMinusUtcNs() == utc);

    // At 29.97 the count restarts at the frame that holds midnight.
    sched::ClockSettings ntsc;
    ntsc.rate = sched::Rate::k2997();
    auto drop = sched::FrameClock::create(ntsc);
    REQUIRE(drop);
    const int64_t midnight = (*drop)->frameAt(kNewYear2026 * kSecond + (*drop)->taiMinusUtcNs() - 1);
    CHECK((*drop)->timecodeOf(midnight + 1) == tc(0, 0, 0, 0, true));
    CHECK((*drop)->timecodeOf(midnight + 1 + 17982 * 6) == tc(1, 0, 0, 0, true));
}

TEST_CASE("running free, frames are waited for to within a millisecond", "[sched][timing]") {
    auto clock = sched::FrameClock::create({});
    REQUIRE(clock);
    int64_t worst = 0;
    const int64_t first = (*clock)->frameAt((*clock)->nowTaiNs()) + 1;
    for (int64_t k = 0; k < 10; ++k) {
        const sched::Tick tick = (*clock)->waitFor(first + k);
        CHECK(tick.lateNs() >= 0);
        worst = std::max(worst, tick.lateNs());
    }
    std::printf("  free run, 10 frames at 25: latest wake %.3f ms after its alignment point\n",
                static_cast<double>(worst) / 1e6);
    CHECK(worst < 1'000'000);
}

TEST_CASE("following a PTP master on loopback, the clock reads the master's time", "[sched][ptp][timing]") {
    genlock::PtpClock master;
    const std::string why = master.startMaster(0);
    if (!why.empty()) {
        SKIP("no PTP master on loopback: " << why);
    }
    sched::ClockSettings settings;
    settings.ptpMaster = "127.0.0.1";
    settings.port = master.boundPort();
    auto clock = sched::FrameClock::create(settings);
    if (!clock) FAIL(clock.error().toString());
    REQUIRE((*clock)->following());
    REQUIRE((*clock)->waitForLock(std::chrono::seconds(5)));
    // Let the filter see a few more cycles than the first.
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while ((*clock)->lock().samples < 8 && std::chrono::steady_clock::now() < until) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const int64_t masterTai = static_cast<int64_t>(master.masterNowNs()) + (*clock)->taiMinusUtcNs();
    const int64_t error = (*clock)->nowTaiNs() - masterTai;
    const auto lock = (*clock)->lock();
    std::printf("  PTP loopback: %llu samples, offset %lld ns, jitter %lld ns, reads %lld ns off the master\n",
                static_cast<unsigned long long>(lock.samples), static_cast<long long>(lock.offsetNs),
                static_cast<long long>(lock.jitterNs), static_cast<long long>(error));
    CHECK(std::llabs(error) < 1'000'000);
    const sched::Tick tick = (*clock)->waitFor((*clock)->frameAt((*clock)->nowTaiNs()) + 2);
    CHECK(tick.lateNs() < 1'000'000);
}

TEST_CASE("an EXR carries its timecode and rate as OpenEXR attributes", "[sched][exr]") {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "lrt_timecode.exr";
    const std::vector<float> rgba(4 * 4, 0.5F);
    const std::vector<io::ExrAttribute> attributes{
        io::ExrAttribute::timecode("timeCode", sched::packTimecode(tc(10, 20, 30, 12, true))),
        io::ExrAttribute::rational("framesPerSecond", 30000, 1001),
        io::ExrAttribute::text("lrt:taiNs", "1767225637000000000"),
    };
    REQUIRE(io::writeExr(path, 2, 2, rgba, {}, true, attributes));
    auto read = io::readExr(path);
    REQUIRE(read);
    const auto find = [&](const std::string& name) -> const io::ExrAttribute* {
        for (const auto& a : read->attributes) {
            if (a.name == name) return &a;
        }
        return nullptr;
    };
    for (const io::ExrAttribute& written : attributes) {
        const io::ExrAttribute* back = find(written.name);
        REQUIRE(back != nullptr);
        CHECK(back->type == written.type);
        CHECK(back->value == written.value);
    }
    std::filesystem::remove(path);
}
