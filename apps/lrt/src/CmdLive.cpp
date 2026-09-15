// Copyright (c) 2026 lucabRTrender contributors.
//
// `lrt live`: a USD stage rendered frame by frame on a clock. Frame N is drawn
// for the instant ST 2059-1 says frame N begins, so two render nodes following
// one PTP master draw the same frame for the same instant; each image carries
// its timecode and TAI time.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>

#include "Commands.h"
#include "lrt/io/Exr.h"
#include "lrt/sched/FrameClock.h"
#include "lrt/usd/StageRenderer.h"

namespace lrt::cli {
namespace {

bool parseRate(const std::string& text, sched::Rate& rate) {
    if (text == "23.976" || text == "23.98") {
        rate = {24000, 1001};
    } else if (text == "29.97") {
        rate = sched::Rate::k2997();
    } else if (text == "59.94") {
        rate = sched::Rate::k5994();
    } else if (long long n = 0, d = 0; std::sscanf(text.c_str(), "%lld/%lld", &n, &d) == 2) {
        rate = {n, d};
    } else if (std::sscanf(text.c_str(), "%lld", &n) == 1 && text.find('.') == std::string::npos) {
        rate = {n, 1};
    } else {
        return false;
    }
    return rate.valid();
}

std::string timecodeText(const sched::Timecode& tc) {
    char text[16];
    std::snprintf(text, sizeof text, "%02d:%02d:%02d%c%02d", tc.hours, tc.minutes, tc.seconds,
                  tc.dropFrame ? ';' : ':', tc.frames);
    return text;
}

/// `name.####.exr` with the #s as a zero-padded number; no #s, the number
/// goes before the extension.
std::string framePath(const std::string& pattern, int64_t number) {
    const size_t first = pattern.find('#');
    if (first == std::string::npos) {
        const size_t dot = pattern.rfind('.');
        const std::string n = "." + std::to_string(number);
        return dot == std::string::npos ? pattern + n : pattern.substr(0, dot) + n + pattern.substr(dot);
    }
    const size_t last = pattern.find_first_not_of('#', first);
    const size_t width = (last == std::string::npos ? pattern.size() : last) - first;
    std::string digits = std::to_string(number);
    if (digits.size() < width) {
        digits.insert(0, width - digits.size(), '0');
    }
    return pattern.substr(0, first) + digits + (last == std::string::npos ? "" : pattern.substr(last));
}

/// EXRs written on their own thread: compressing one takes longer than a
/// frame's render at some sizes, and the clock does not wait for disks.
class Writer {
public:
    struct Job {
        std::string                  path;
        uint32_t                     width = 0, height = 0;
        usd::StageImage              image;
        std::vector<io::ExrAttribute> attributes;
    };

    Writer() : thread_([this] { run(); }) {}
    ~Writer() { finish(); }

    void push(Job job) {
        {
            std::lock_guard lock(mutex_);
            jobs_.push_back(std::move(job));
        }
        wake_.notify_one();
    }
    /// Waits for every queued image; the first failure, if any.
    std::string finish() {
        {
            std::lock_guard lock(mutex_);
            done_ = true;
        }
        wake_.notify_one();
        if (thread_.joinable()) {
            thread_.join();
        }
        return error_;
    }

private:
    void run() {
        std::unique_lock lock(mutex_);
        for (;;) {
            wake_.wait(lock, [&] { return done_ || !jobs_.empty(); });
            if (jobs_.empty()) {
                return;
            }
            Job job = std::move(jobs_.front());
            jobs_.pop_front();
            lock.unlock();
            auto saved = io::writeExr(job.path, job.width, job.height, job.image.rgba, job.image.depth, true,
                                      job.attributes);
            lock.lock();
            if (!saved && error_.empty()) {
                error_ = saved.error().toString();
            }
        }
    }

    std::mutex              mutex_;
    std::condition_variable wake_;
    std::deque<Job>         jobs_;
    bool                    done_ = false;
    std::string             error_;
    std::thread             thread_;
};

}   // namespace

void addLive(CLI::App& app) {
    struct Options {
        std::string stage, camera, output = "live.####.exr", size = "1920x1080", technique = "raster";
        std::string rate = "25", ptp;
        uint16_t    port = genlock::PtpClock::kDefaultPort;
        unsigned    domain = 0;
        double      lockTimeout = 10.0;
        int64_t     frames = 25;
        double      start = std::numeric_limits<double>::quiet_NaN();
        std::string at;
        int64_t     taiMinusUtc = genlock::Alignment::kDefaultTaiMinusUtcSeconds;
    };
    auto o = std::make_shared<Options>();
    auto* cmd = app.add_subcommand("live", "render a USD stage on a clock: each frame at its ST 2059-1 instant");
    cmd->add_option("stage", o->stage, ".usd / .usda / .usdc")->required();
    cmd->add_option("--camera", o->camera, "camera prim path (default: the first)");
    cmd->add_option("--size", o->size, "WIDTHxHEIGHT");
    cmd->add_option("--technique", o->technique, "raster | rt");
    cmd->add_option("--rate", o->rate, "frames per second: 25, 50, 29.97, 59.94, 23.976 or N/D");
    cmd->add_option("--ptp", o->ptp, "follow this PTP master (genlock-cli master); default: this machine's clock");
    cmd->add_option("--port", o->port, "PTP port (319 needs privileges; any port both ends agree on)");
    cmd->add_option("--domain", o->domain, "PTP domain");
    cmd->add_option("--lock-timeout", o->lockTimeout, "seconds to wait for a PTP lock");
    cmd->add_option("--frames", o->frames, "frames to write");
    cmd->add_option("--start", o->start, "USD time that plays first (default: the stage's start)");
    cmd->add_option("--at", o->at,
                    "HH:MM:SS:FF (UTC time of day) at which --start plays: nodes given the same one draw the "
                    "same time for the same instant (default: when this node begins)");
    cmd->add_option("--tai-utc", o->taiMinusUtc, "TAI - UTC in seconds, for timecodes");
    cmd->add_option("-o,--output", o->output, "EXR path; #s become the frame number");
    cmd->callback([o] {
        const auto fail = [](const std::string& why) {
            std::fprintf(stderr, "%s\n", why.c_str());
            throw CLI::RuntimeError(1);
        };
        uint32_t width = 0, height = 0;
        if (std::sscanf(o->size.c_str(), "%ux%u", &width, &height) != 2 || width == 0 || height == 0) {
            fail("--size wants WIDTHxHEIGHT");
        }
        sched::ClockSettings settings;
        if (!parseRate(o->rate, settings.rate)) {
            fail("--rate wants 25, 29.97, 60000/1001 and the like");
        }
        settings.ptpMaster = o->ptp;
        settings.port = o->port;
        settings.domain = static_cast<uint8_t>(o->domain);
        settings.taiMinusUtcSeconds = o->taiMinusUtc;
        auto clock = sched::FrameClock::create(settings);
        if (!clock) fail(clock.error().toString());
        if ((*clock)->following()) {
            std::printf("following PTP master %s:%u...\n", o->ptp.c_str(), o->port);
            if (auto locked = (*clock)->waitForLock(std::chrono::milliseconds(
                    static_cast<int64_t>(o->lockTimeout * 1000.0)));
                !locked) {
                fail(locked.error().toString());
            }
            const auto lock = (*clock)->lock();
            std::printf("locked: offset %.3f ms, path delay %.3f ms, jitter %.3f ms\n",
                        static_cast<double>(lock.offsetNs) / 1e6, static_cast<double>(lock.meanPathDelayNs) / 1e6,
                        static_cast<double>(lock.jitterNs) / 1e6);
        }
        auto renderer = usd::StageRenderer::open(o->stage);
        if (!renderer) fail(renderer.error().toString());
        const sched::Rate rate = (*clock)->rate();
        const double start = std::isnan(o->start) ? (*renderer)->startTimeCode() : o->start;
        const double codesPerFrame = (*renderer)->timeCodesPerSecond() * static_cast<double>(rate.denominator) /
                                     static_cast<double>(rate.numerator);

        // A stage without cameras is framed as lrt stage and lrt view frame it.
        std::optional<render::Camera> own;
        if (o->camera.empty() && (*renderer)->cameras().empty()) {
            auto framed = (*renderer)->framingCamera(start, 35.0, o->technique);
            if (!framed) fail(framed.error().toString());
            own = *framed;
        }
        const auto renderAt = [&](double at) {
            return own ? (*renderer)->render(*own, at, width, height, o->technique)
                       : (*renderer)->render(o->camera, at, width, height, o->technique);
        };
        // A frame before the clock counts: the first render loads the stage
        // onto the device and compiles its shaders, which takes frames.
        if (auto warm = renderAt(start); !warm) {
            fail(warm.error().toString());
        }
        const int64_t now = (*clock)->frameAt((*clock)->nowTaiNs()) + 1;
        int64_t anchor = now;
        if (!o->at.empty()) {
            sched::Timecode at;
            if (!sched::parseTimecode(o->at, at)) {
                fail("--at wants HH:MM:SS:FF");
            }
            anchor = (*clock)->frameLabelled(at, now);
        }
        // Frames before the anchor have no stage time: wait for it.
        const int64_t first = std::max(now, anchor);
        int64_t index = first;
        int64_t dropped = 0;
        Writer writer;
        int64_t worstLate = 0;
        for (int64_t written = 0; written < o->frames; ++written) {
            const sched::Tick tick = (*clock)->waitFor(index);
            worstLate = std::max(worstLate, tick.lateNs());
            const double time = start + static_cast<double>(index - anchor) * codesPerFrame;
            const auto began = std::chrono::steady_clock::now();
            auto image = renderAt(time);
            if (!image) fail(image.error().toString());
            const double renderMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count();

            const sched::Timecode tc = (*clock)->timecodeOf(index);
            const std::vector<io::ExrAttribute> attributes{
                io::ExrAttribute::timecode("timeCode", sched::packTimecode(tc)),
                io::ExrAttribute::rational("framesPerSecond", static_cast<int32_t>(rate.numerator),
                                           static_cast<uint32_t>(rate.denominator)),
                io::ExrAttribute::text("lrt:taiNs", std::to_string(tick.startTaiNs)),
                io::ExrAttribute::text("lrt:frameIndex", std::to_string(index)),
                io::ExrAttribute::float64("lrt:usdTime", time),
                io::ExrAttribute::float64("lrt:wakeLateMs", static_cast<double>(tick.lateNs()) / 1e6),
                io::ExrAttribute::text("lrt:clock", (*clock)->following() ? "ptp " + o->ptp : "free run"),
            };
            // Numbered by clock frames from the start, not by USD time: a
            // stage's time codes need not be the clock's frames.
            const std::string path = framePath(o->output, std::llround(start) + (index - anchor));
            writer.push({path, width, height, std::move(*image), attributes});
            const auto lock = (*clock)->lock();
            std::printf("frame %lld  %s  usd %.3f  woke %+.3f ms  render %.1f ms%s  %s\n",
                        static_cast<long long>(index), timecodeText(tc).c_str(), time,
                        static_cast<double>(tick.lateNs()) / 1e6, renderMs,
                        (*clock)->following()
                            ? ("  jitter " + std::to_string(lock.jitterNs / 1000) + " us").c_str()
                            : "",
                        path.c_str());
            // Frames whose instant passed while this one rendered are not drawn late.
            const int64_t next = std::max(index + 1, (*clock)->frameAt((*clock)->nowTaiNs()) + 1);
            dropped += next - index - 1;
            index = next;
        }
        if (const std::string why = writer.finish(); !why.empty()) {
            fail(why);
        }
        std::printf("%lld frames written, %lld skipped for time, latest wake %.3f ms after its instant\n",
                    static_cast<long long>(o->frames), static_cast<long long>(dropped),
                    static_cast<double>(worstLate) / 1e6);
    });
}

}   // namespace lrt::cli
