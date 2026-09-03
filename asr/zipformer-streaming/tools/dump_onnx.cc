// 列出 onnx 模型的输入/输出名称、形状和自定义元数据（chunk 配置）
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <model.onnx> [...]\n";
        return 1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "dump_onnx");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    for (int a = 1; a < argc; ++a) {
        try {
            Ort::Session s(env, argv[a], so);
            Ort::AllocatorWithDefaultOptions alloc;

            std::cout << "=== " << argv[a] << " ===\n";

            auto meta = s.GetModelMetadata();
            auto keys = meta.GetCustomMetadataMapKeysAllocated(alloc);
            for (size_t i = 0; i < keys.size(); ++i) {
                auto v = meta.LookupCustomMetadataMapAllocated(keys[i].get(), alloc);
                std::cout << "  meta: " << keys[i].get() << " = " << v.get() << "\n";
            }

            auto dump = [&](bool input) {
                size_t n = input ? s.GetInputCount() : s.GetOutputCount();
                for (size_t i = 0; i < n; ++i) {
                    auto name = input ? s.GetInputNameAllocated(i, alloc)
                                      : s.GetOutputNameAllocated(i, alloc);
                    auto info = input ? s.GetInputTypeInfo(i) : s.GetOutputTypeInfo(i);
                    auto ti = info.GetTensorTypeAndShapeInfo();
                    std::string sh;
                    for (auto d : ti.GetShape()) sh += std::to_string(d) + " ";
                    std::cout << "  " << (input ? "IN " : "OUT") << " " << name.get()
                              << " [" << sh << "] elem_type=" << ti.GetElementType()
                              << "\n";
                }
            };
            dump(true);
            dump(false);
        } catch (const std::exception& e) {
            std::cout << "=== " << argv[a] << " === ERROR: " << e.what() << "\n";
        }
    }
    return 0;
}
