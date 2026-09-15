// Copyright (c) 2026 lucabRTrender contributors.
//
// A pass-through scene index the delegate registers for itself, so that it
// sits in every chain a host builds for this renderer. It changes no prim;
// it exists to send notices the scene does not carry -- a camera's shutter
// changed, and every transform and primvar sampled about the old one must
// be sampled again. The change tracker's marks cannot say that: under scene
// index emulation they do not reach prims a scene index owns, and without
// emulation (UsdImagingGLEngine) they are refused outright.
#pragma once

#include <pxr/imaging/hd/filteringSceneIndex.h>

PXR_NAMESPACE_OPEN_SCOPE

TF_DECLARE_REF_PTRS(HdLrtResampleSceneIndex);

class HdLrtResampleSceneIndex final : public HdSingleInputFilteringSceneIndexBase {
public:
    static HdLrtResampleSceneIndexRefPtr New(const HdSceneIndexBaseRefPtr& input);

    /// Every prim dirtied at `locators`.
    void DirtyAll(const HdDataSourceLocatorSet& locators);

    HdSceneIndexPrim GetPrim(const SdfPath& primPath) const override;
    SdfPathVector GetChildPrimPaths(const SdfPath& primPath) const override;

protected:
    explicit HdLrtResampleSceneIndex(const HdSceneIndexBaseRefPtr& input);

    void _PrimsAdded(const HdSceneIndexBase& sender, const HdSceneIndexObserver::AddedPrimEntries& entries) override;
    void _PrimsRemoved(const HdSceneIndexBase& sender, const HdSceneIndexObserver::RemovedPrimEntries& entries) override;
    void _PrimsDirtied(const HdSceneIndexBase& sender, const HdSceneIndexObserver::DirtiedPrimEntries& entries) override;
};

/// Every HdLrtResampleSceneIndex upstream of `terminal`, found by walking its
/// inputs: dirtied at `locators`. Returns how many were found.
size_t HdLrtResampleUpstream(const HdSceneIndexBaseRefPtr& terminal, const HdDataSourceLocatorSet& locators);

PXR_NAMESPACE_CLOSE_SCOPE
