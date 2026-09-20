#pragma once

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "yolov10_detector.h"

// Collapses the 34-class HaGRID gesture space down to the game's gesture set
// (rock/paper/scissors). The mapping comes from config.json "rps_map":
//   { "Rock": ["fist"], "Paper": ["palm","stop"], ... }
// i.e. label -> model class names. After build(), detections of classes
// outside the mapping are dropped and the rest carry the game label.
class RpsMapper {
public:
    using LabelMap = std::map<std::string, std::vector<std::string>>;

    struct Label {
        std::string text;
        int color_index = 0;  // index into the drawing palette, one per label
    };

    // Resolve every source class name against class_names (model id order).
    // Throws std::invalid_argument on unknown names, empty labels/lists, or a
    // source class claimed by two labels: a typo must fail loudly instead of
    // silently relabeling the wrong gesture.
    void build(const LabelMap& label_map, const std::vector<std::string>& class_names) {
        label_of_.clear();
        allowed_.clear();
        int color_index = 0;
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
                label_of_[id] = {label, color_index};
            }
            ++color_index;
        }
    }

    bool enabled() const { return !label_of_.empty(); }

    // Drop detections whose class is not part of the mapping (in place).
    void filter(std::vector<Detection>& detections) const {
        if (!enabled()) return;
        detections.erase(std::remove_if(detections.begin(), detections.end(),
                                        [&](const Detection& d) {
                                            return allowed_.find(d.class_id) == allowed_.end();
                                        }),
                         detections.end());
    }

    // Display label for each detection; call after filter() so every entry
    // resolves. Returns an empty vector when the mapper is disabled.
    std::vector<Label> labels(const std::vector<Detection>& detections) const {
        std::vector<Label> out;
        if (!enabled()) return out;
        out.reserve(detections.size());
        for (const Detection& d : detections) {
            const auto it = label_of_.find(d.class_id);
            if (it != label_of_.end()) out.push_back(it->second);
        }
        return out;
    }

private:
    std::unordered_map<int, Label> label_of_;
    std::unordered_set<int> allowed_;
};
