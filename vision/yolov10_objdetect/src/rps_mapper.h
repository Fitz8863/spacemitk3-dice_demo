#pragma once

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "yolov10_detector.h"

// Collapses the model's gesture vocabulary onto the game's labels via the
// config "rps_map" table: { "Rock": ["fist", ...], ... } i.e. game label ->
// model class names. build() resolves every source name against the
// config-declared classes; fold() then drops unmapped detections and
// rewrites each surviving class_id to the label's declaration index, so
// every downstream consumer (stability signature, drawing, observation
// events) works in the folded label space. The package itself has no
// built-in vocabulary: classes and the fold table always come from config.
class RpsMapper {
public:
    using LabelMap = std::map<std::string, std::vector<std::string>>;

    // Resolve every source class name against class_names (model id order).
    // Throws std::invalid_argument on unknown names, empty labels/lists, or a
    // source class claimed by two labels: a typo must fail loudly instead of
    // silently relabeling the wrong gesture.
    void build(const LabelMap& label_map, const std::vector<std::string>& class_names) {
        label_of_.clear();
        allowed_.clear();
        folded_labels_.clear();
        int folded_id = 0;
        for (const auto& [label, source_names] : label_map) {
            if (label.empty()) throw std::invalid_argument("rps_map contains an empty label");
            if (source_names.empty())
                throw std::invalid_argument("rps_map label '" + label + "' has no source classes");
            for (const std::string& name : source_names) {
                const auto it = std::find(class_names.begin(), class_names.end(), name);
                if (it == class_names.end())
                    throw std::invalid_argument("rps_map references unknown class '" + name + "'");
                const int id = static_cast<int>(it - class_names.begin());
                if (!allowed_.insert(id).second)
                    throw std::invalid_argument("rps_map maps class '" + name + "' to two labels");
                label_of_[id] = folded_id;
            }
            folded_labels_.push_back(label);
            ++folded_id;
        }
    }

    bool enabled() const { return !label_of_.empty(); }

    // Game label text for a folded class id (the rps_map declaration index).
    const std::string& folded_label(int folded_id) const {
        static const std::string kUnknown = "?";
        if (folded_id < 0 || folded_id >= static_cast<int>(folded_labels_.size()))
            return kUnknown;
        return folded_labels_[static_cast<std::size_t>(folded_id)];
    }

    // Drop detections whose class is not part of the mapping and rewrite the
    // surviving class_id to the folded label index (in place). Must run
    // before the stability signature is computed: source-class flicker
    // within one game label (palm -> stop, both "Paper") must not reset the
    // stability streak.
    void fold(std::vector<Detection>& detections) const {
        if (!enabled()) return;
        std::vector<Detection> out;
        out.reserve(detections.size());
        for (const Detection& d : detections) {
            const auto it = label_of_.find(d.class_id);
            if (it == label_of_.end()) continue;
            Detection folded = d;
            folded.class_id = it->second;
            out.push_back(std::move(folded));
        }
        detections = std::move(out);
    }

private:
    std::unordered_map<int, int> label_of_;   // source class id -> folded id
    std::vector<std::string> folded_labels_;  // folded id -> game label text
    std::unordered_set<int> allowed_;         // source ids accepted by the map
};
