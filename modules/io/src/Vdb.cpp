// Copyright (c) 2026 lucabRTrender contributors.

// NanoVDB 32.x's GridBuilder still names std::result_of, which C++20
// removed; libc++ keeps it behind this macro (libstdc++ keeps it as is).
#define _LIBCPP_ENABLE_CXX20_REMOVED_TYPE_TRAITS 1
#include "lrt/io/Vdb.h"

#include <cstring>
#include <mutex>

#if defined(LRT_HAVE_OPENVDB)
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridChecksum.h>
#include <nanovdb/util/GridStats.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#endif

namespace lrt::io {

bool haveOpenVdb() noexcept {
#if defined(LRT_HAVE_OPENVDB)
    return true;
#else
    return false;
#endif
}

#if defined(LRT_HAVE_OPENVDB)

namespace {

void initialiseOnce() {
    static std::once_flag once;
    std::call_once(once, [] { openvdb::initialize(); });
}

}   // namespace

Result<std::vector<std::string>> vdbGridNames(const std::filesystem::path& path) {
    initialiseOnce();
    openvdb::io::File file(path.string());
    try {
        file.open(false);
        std::vector<std::string> names;
        for (auto it = file.beginName(); it != file.endName(); ++it) {
            names.push_back(it.gridName());
        }
        file.close();
        return names;
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::IoFailure, "cannot read '{}': {}", path.string(), e.what());
    }
}

Result<NanoGrid> readVdbGrid(const std::filesystem::path& path, const std::string& name) {
    initialiseOnce();
    openvdb::io::File file(path.string());
    openvdb::GridBase::Ptr base;
    try {
        file.open(false);
        std::string wanted = name;
        if (wanted.empty()) {
            if (file.beginName() == file.endName()) {
                return Error::make(ErrorCode::NotFound, "'{}' holds no grid", path.string());
            }
            wanted = file.beginName().gridName();
        }
        if (!file.hasGrid(wanted)) {
            return Error::make(ErrorCode::NotFound, "'{}' has no grid '{}'", path.string(), wanted);
        }
        base = file.readGrid(wanted);
        file.close();
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::IoFailure, "cannot read '{}': {}", path.string(), e.what());
    }
    auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base);
    if (!grid) {
        return Error::make(ErrorCode::Unsupported, "grid '{}' of '{}' is {}, not float", base->getName(),
                           path.string(), base->valueType());
    }
    if (!grid->transform().isLinear()) {
        return Error::make(ErrorCode::Unsupported, "grid '{}': only linear transforms are read", base->getName());
    }
    // Tiles -- a constant region OpenVDB keeps as one value at an internal
    // node -- become leaves, so the tree the kernels walk is leaves alone:
    // the bounds, the maxima and the majorant are taken per leaf. A
    // re-layout, not a statistic.
    grid->tree().voxelizeActiveTiles();
    // The re-layout, with every statistic off: no bounding boxes, extrema
    // or means are computed here, and no checksum either.
    nanovdb::GridHandle<nanovdb::HostBuffer> handle;
    try {
        handle = nanovdb::openToNanoVDB(*grid, nanovdb::StatsMode::Disable, nanovdb::ChecksumMode::Disable);
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::IoFailure, "grid '{}': cannot lay out as NanoVDB: {}", base->getName(),
                           e.what());
    }
    NanoGrid out;
    out.name = base->getName();
    out.words.resize((handle.size() + 3) / 4, 0u);
    std::memcpy(out.words.data(), handle.data(), handle.size());
    // The header's leaf count: bookkeeping the kernels index by.
    out.leafCount = handle.gridMetaData() != nullptr ? handle.gridMetaData()->nodeCount(0) : 0;
    out.voxelSize = grid->transform().voxelSize()[0];
    const openvdb::math::Mat4d m = grid->transform().baseMap()->getAffineMap()->getMat4();
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out.indexToWorld[static_cast<size_t>(r * 4 + c)] = m(r, c);
        }
    }
    return out;
}

Result<void> writeVdbBoxes(const std::filesystem::path& path, const std::string& name, double voxelSize,
                           std::span<const VdbBox> boxes) {
    initialiseOnce();
    try {
        openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.0F);
        grid->setName(name);
        grid->setGridClass(openvdb::GRID_FOG_VOLUME);
        grid->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));
        for (const VdbBox& box : boxes) {
            const openvdb::CoordBBox bbox(openvdb::Coord(box.min[0], box.min[1], box.min[2]),
                                          openvdb::Coord(box.max[0] - 1, box.max[1] - 1, box.max[2] - 1));
            grid->fill(bbox, box.value, true);
        }
        openvdb::io::File file(path.string());
        openvdb::GridPtrVec grids{grid};
        file.write(grids);
        file.close();
        return ok();
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::IoFailure, "cannot write '{}': {}", path.string(), e.what());
    }
}

#else

Result<std::vector<std::string>> vdbGridNames(const std::filesystem::path& path) {
    return Error::make(ErrorCode::Unsupported, "'{}': built without OpenVDB", path.string());
}

Result<NanoGrid> readVdbGrid(const std::filesystem::path& path, const std::string&) {
    return Error::make(ErrorCode::Unsupported, "'{}': built without OpenVDB", path.string());
}

Result<void> writeVdbBoxes(const std::filesystem::path& path, const std::string&, double, std::span<const VdbBox>) {
    return Error::make(ErrorCode::Unsupported, "'{}': built without OpenVDB", path.string());
}

#endif

}   // namespace lrt::io
