// Copyright (c) 2026 lucabRTrender contributors.
//
// Material binding purposes a render settings prim can change between
// frames. HdsiMaterialBindingResolvingSceneIndex resolves a prim's bindings
// by a list of purposes fixed when it is made, and a filter cannot be
// swapped out of a chain the render index already observes; this one holds
// its list and, when the list changes, tells every prim with bindings that
// they are dirty -- the notice meshes sync their material ids from.
#pragma once

#include <pxr/imaging/hd/filteringSceneIndex.h>

PXR_NAMESPACE_OPEN_SCOPE

TF_DECLARE_REF_PTRS(HdLrtBindingPurposesSceneIndex);

class HdLrtBindingPurposesSceneIndex final : public HdSingleInputFilteringSceneIndexBase {
public:
    /// Each prim's bindings become one all-purpose binding: the first of
    /// `purposes` the prim has ("" is the all-purpose binding itself).
    static HdLrtBindingPurposesSceneIndexRefPtr New(const HdSceneIndexBaseRefPtr& input, const TfTokenVector& purposes);

    /// A new list; every prim with bindings is dirtied when it differs.
    void SetPurposes(const TfTokenVector& purposes);

    /// Every prim dirtied at `locators`: what a change the scene does not
    /// carry needs -- a shutter, which every sampled transform and primvar
    /// depends on. (The change tracker's own marks do not reach prims a
    /// scene index owns under emulation.)
    void DirtyAll(const HdDataSourceLocatorSet& locators);
    [[nodiscard]] const TfTokenVector& GetPurposes() const { return _purposes; }

    HdSceneIndexPrim GetPrim(const SdfPath& primPath) const override;
    SdfPathVector GetChildPrimPaths(const SdfPath& primPath) const override;

protected:
    HdLrtBindingPurposesSceneIndex(const HdSceneIndexBaseRefPtr& input, const TfTokenVector& purposes);

    void _PrimsAdded(const HdSceneIndexBase& sender, const HdSceneIndexObserver::AddedPrimEntries& entries) override;
    void _PrimsRemoved(const HdSceneIndexBase& sender, const HdSceneIndexObserver::RemovedPrimEntries& entries) override;
    void _PrimsDirtied(const HdSceneIndexBase& sender, const HdSceneIndexObserver::DirtiedPrimEntries& entries) override;

private:
    TfTokenVector _purposes;
};

PXR_NAMESPACE_CLOSE_SCOPE
