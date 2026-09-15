// Copyright (c) 2026 lucabRTrender contributors.
#include "BindingPurposes.h"

#include <pxr/imaging/hd/materialBindingsSchema.h>
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/sceneIndexPrimView.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

/// A prim whose bindings are the one its purposes pick, under the
/// all-purpose name every purpose falls back to.
class ResolvedPrim final : public HdContainerDataSource {
public:
    HD_DECLARE_DATASOURCE(ResolvedPrim);

    ResolvedPrim(const HdContainerDataSourceHandle& input, const TfTokenVector& purposes)
        : _input(input), _purposes(purposes) {}

    TfTokenVector GetNames() override { return _input->GetNames(); }

    HdDataSourceBaseHandle Get(const TfToken& name) override {
        HdDataSourceBaseHandle data = _input->Get(name);
        if (name != HdMaterialBindingsSchema::GetSchemaToken()) {
            return data;
        }
        const HdContainerDataSourceHandle bindings = HdContainerDataSource::Cast(data);
        if (!bindings) {
            return data;
        }
        for (const TfToken& purpose : _purposes) {
            if (HdDataSourceBaseHandle binding = bindings->Get(purpose)) {
                return HdRetainedContainerDataSource::New(HdMaterialBindingsSchemaTokens->allPurpose, binding);
            }
        }
        return HdRetainedContainerDataSource::New();
    }

private:
    HdContainerDataSourceHandle _input;
    TfTokenVector               _purposes;
};

}   // namespace

HdLrtBindingPurposesSceneIndexRefPtr HdLrtBindingPurposesSceneIndex::New(const HdSceneIndexBaseRefPtr& input,
                                                                         const TfTokenVector& purposes) {
    return TfCreateRefPtr(new HdLrtBindingPurposesSceneIndex(input, purposes));
}

HdLrtBindingPurposesSceneIndex::HdLrtBindingPurposesSceneIndex(const HdSceneIndexBaseRefPtr& input,
                                                               const TfTokenVector& purposes)
    : HdSingleInputFilteringSceneIndexBase(input), _purposes(purposes) {}

void HdLrtBindingPurposesSceneIndex::SetPurposes(const TfTokenVector& purposes) {
    if (purposes == _purposes) {
        return;
    }
    _purposes = purposes;
    const HdSceneIndexBaseRefPtr input = _GetInputSceneIndex();
    if (!input) {
        return;
    }
    // Walking the scene's paths is bookkeeping: which prims carry bindings.
    HdSceneIndexObserver::DirtiedPrimEntries dirtied;
    for (const SdfPath& path : HdSceneIndexPrimView(input)) {
        const HdSceneIndexPrim prim = input->GetPrim(path);
        if (prim.dataSource && prim.dataSource->Get(HdMaterialBindingsSchema::GetSchemaToken())) {
            dirtied.emplace_back(path, HdMaterialBindingsSchema::GetDefaultLocator());
        }
    }
    if (!dirtied.empty()) {
        _SendPrimsDirtied(dirtied);
    }
}

void HdLrtBindingPurposesSceneIndex::DirtyAll(const HdDataSourceLocatorSet& locators) {
    const HdSceneIndexBaseRefPtr input = _GetInputSceneIndex();
    if (!input) {
        return;
    }
    HdSceneIndexObserver::DirtiedPrimEntries dirtied;
    for (const SdfPath& path : HdSceneIndexPrimView(input)) {
        dirtied.emplace_back(path, locators);
    }
    if (!dirtied.empty()) {
        _SendPrimsDirtied(dirtied);
    }
}

HdSceneIndexPrim HdLrtBindingPurposesSceneIndex::GetPrim(const SdfPath& primPath) const {
    const HdSceneIndexBaseRefPtr input = _GetInputSceneIndex();
    if (!input) {
        return {TfToken(), nullptr};
    }
    HdSceneIndexPrim prim = input->GetPrim(primPath);
    if (prim.dataSource) {
        prim.dataSource = ResolvedPrim::New(prim.dataSource, _purposes);
    }
    return prim;
}

SdfPathVector HdLrtBindingPurposesSceneIndex::GetChildPrimPaths(const SdfPath& primPath) const {
    const HdSceneIndexBaseRefPtr input = _GetInputSceneIndex();
    return input ? input->GetChildPrimPaths(primPath) : SdfPathVector();
}

void HdLrtBindingPurposesSceneIndex::_PrimsAdded(const HdSceneIndexBase&,
                                                 const HdSceneIndexObserver::AddedPrimEntries& entries) {
    _SendPrimsAdded(entries);
}

void HdLrtBindingPurposesSceneIndex::_PrimsRemoved(const HdSceneIndexBase&,
                                                   const HdSceneIndexObserver::RemovedPrimEntries& entries) {
    _SendPrimsRemoved(entries);
}

void HdLrtBindingPurposesSceneIndex::_PrimsDirtied(const HdSceneIndexBase&,
                                                   const HdSceneIndexObserver::DirtiedPrimEntries& entries) {
    _SendPrimsDirtied(entries);
}

PXR_NAMESPACE_CLOSE_SCOPE
