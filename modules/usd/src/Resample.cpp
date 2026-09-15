// Copyright (c) 2026 lucabRTrender contributors.
#include "Resample.h"

#include <set>
#include <vector>

#include <pxr/imaging/hd/sceneIndexPrimView.h>

PXR_NAMESPACE_OPEN_SCOPE

HdLrtResampleSceneIndexRefPtr HdLrtResampleSceneIndex::New(const HdSceneIndexBaseRefPtr& input) {
    return TfCreateRefPtr(new HdLrtResampleSceneIndex(input));
}

HdLrtResampleSceneIndex::HdLrtResampleSceneIndex(const HdSceneIndexBaseRefPtr& input)
    : HdSingleInputFilteringSceneIndexBase(input) {}

void HdLrtResampleSceneIndex::DirtyAll(const HdDataSourceLocatorSet& locators) {
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

HdSceneIndexPrim HdLrtResampleSceneIndex::GetPrim(const SdfPath& primPath) const {
    const HdSceneIndexBaseRefPtr input = _GetInputSceneIndex();
    return input ? input->GetPrim(primPath) : HdSceneIndexPrim{TfToken(), nullptr};
}

SdfPathVector HdLrtResampleSceneIndex::GetChildPrimPaths(const SdfPath& primPath) const {
    const HdSceneIndexBaseRefPtr input = _GetInputSceneIndex();
    return input ? input->GetChildPrimPaths(primPath) : SdfPathVector();
}

void HdLrtResampleSceneIndex::_PrimsAdded(const HdSceneIndexBase&, const HdSceneIndexObserver::AddedPrimEntries& entries) {
    _SendPrimsAdded(entries);
}

void HdLrtResampleSceneIndex::_PrimsRemoved(const HdSceneIndexBase&,
                                            const HdSceneIndexObserver::RemovedPrimEntries& entries) {
    _SendPrimsRemoved(entries);
}

void HdLrtResampleSceneIndex::_PrimsDirtied(const HdSceneIndexBase&,
                                            const HdSceneIndexObserver::DirtiedPrimEntries& entries) {
    _SendPrimsDirtied(entries);
}

size_t HdLrtResampleUpstream(const HdSceneIndexBaseRefPtr& terminal, const HdDataSourceLocatorSet& locators) {
    std::vector<HdSceneIndexBaseRefPtr> pending{terminal};
    std::set<const HdSceneIndexBase*> seen;
    std::vector<HdLrtResampleSceneIndexRefPtr> found;
    while (!pending.empty()) {
        const HdSceneIndexBaseRefPtr scene = pending.back();
        pending.pop_back();
        if (!scene || !seen.insert(get_pointer(scene)).second) {
            continue;
        }
        if (auto resample = TfDynamic_cast<HdLrtResampleSceneIndexRefPtr>(scene)) {
            found.push_back(resample);
        }
        if (auto filtering = TfDynamic_cast<HdFilteringSceneIndexBaseRefPtr>(scene)) {
            for (const HdSceneIndexBaseRefPtr& input : filtering->GetInputScenes()) {
                pending.push_back(input);
            }
        }
        if (HdEncapsulatingSceneIndexBase* encapsulating = HdEncapsulatingSceneIndexBase::Cast(scene)) {
            for (const HdSceneIndexBaseRefPtr& inner : encapsulating->GetEncapsulatedScenes()) {
                pending.push_back(inner);
            }
        }
    }
    for (const HdLrtResampleSceneIndexRefPtr& resample : found) {
        resample->DirtyAll(locators);
    }
    return found.size();
}

PXR_NAMESPACE_CLOSE_SCOPE
