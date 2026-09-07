# ASR API

离线语音识别框架，支持 C++ 和 Python。

## 功能特性

- **非流式识别**: 文件识别、内存音频识别
- **流式识别**: 边录边识别，支持回调模式
- **Flush 接口**: 用户可控的断句时机，配合 VAD 或定时器实现实时识别
- **多语言**: 中文、英文、日语、韩语、粤语、自动检测
- **自动标点**: 可选的标点符号添加

---

## C++ API

```cpp
namespace SpacemiT {

// =============================================================================
// AsrConfig - 配置
// =============================================================================

struct AsrConfig {
    std::string engine = "sensevoice";  // 引擎类型
    std::string model_dir;              // 模型目录
    std::string language = "zh";        // 语言 ("zh", "en", "ja", "ko", "yue", "auto")
    bool punctuation = true;            // 自动标点
    int sample_rate = 16000;            // 采样率

    static AsrConfig Preset(const std::string& name);
    static std::vector<std::string> AvailablePresets();
};

// =============================================================================
// Sentence - 句子结果
// =============================================================================

struct Sentence {
    std::string text;        // 识别文本
    int begin_time = 0;      // 开始时间 (毫秒)
    int end_time = 0;        // 结束时间 (毫秒)
    float confidence = 0.0f; // 置信度 [0.0, 1.0]
};

// =============================================================================
// RecognitionResult - 识别结果
// =============================================================================

class RecognitionResult {
public:
    Sentence GetSentence() const;              // 主句子
    std::vector<Sentence> GetSentences() const; // 所有句子
    std::string GetText() const;               // 完整文本
    bool IsSentenceEnd() const;                // 是否最终结果
    bool IsEmpty() const;                      // 是否为空
    int GetAudioDuration() const;              // 音频时长 (ms)
    int GetProcessingTime() const;             // 处理时间 (ms)
    float GetRTF() const;                      // 实时率
};

// =============================================================================
// AsrEngineCallback - 流式回调接口
// =============================================================================

/**
 * 回调调用顺序:
 *   Start() -> OnOpen() -> OnEvent()* -> Stop() -> OnComplete() -> OnClose()
 *   错误时: OnOpen() -> ... -> OnError() -> OnClose()
 */
class AsrEngineCallback {
public:
    virtual void OnOpen() {}                                      // 会话开始
    virtual void OnEvent(std::shared_ptr<RecognitionResult>) {}   // 识别结果
    virtual void OnComplete() {}                                  // 识别完成
    virtual void OnError(std::shared_ptr<RecognitionResult>) {}   // 发生错误
    virtual void OnClose() {}                                     // 会话关闭
};

// =============================================================================
// AsrEngine - 识别引擎
// =============================================================================

class AsrEngine {
public:
    explicit AsrEngine(const std::string& engine = "sensevoice",
                       const std::string& model_dir = "");
    explicit AsrEngine(const AsrConfig& config);

    // 非流式识别 (阻塞)
    std::shared_ptr<RecognitionResult> Call(const std::string& file_path);
    std::shared_ptr<RecognitionResult> Recognize(const std::vector<int16_t>& audio, int sample_rate = 16000);
    std::shared_ptr<RecognitionResult> Recognize(const std::vector<float>& audio, int sample_rate = 16000);

    // 流式识别
    void SetCallback(std::shared_ptr<AsrEngineCallback> callback);
    void Start(const std::string& phrase_id = "");
    void SendAudioFrame(const std::vector<uint8_t>& data);  // 16kHz, 16bit, mono, PCM
    void Flush();  // 立即识别当前缓冲区 (不关闭会话)
    void Stop();

    // 动态配置
    void SetLanguage(const std::string& language);
    void SetPunctuation(bool enabled);
    AsrConfig GetConfig() const;

    // 辅助方法
    bool IsInitialized() const;
    std::string GetEngineName() const;
};

}  // namespace SpacemiT
```

### C++ 示例

```cpp
#include "asr_service.h"
using namespace SpacemiT;

int main() {
    // 配置
    AsrConfig config = AsrConfig::Preset("sensevoice");
    config.language = "zh";
    config.punctuation = true;

    // 文件识别
    auto engine = std::make_shared<AsrEngine>(config);
    auto result = engine->Call("audio.wav");
    if (result && !result->IsEmpty()) {
        std::cout << "文本: " << result->GetText() << std::endl;
        std::cout << "RTF: " << result->GetRTF() << std::endl;
    }

    // 内存识别 (音频数据需用户自行加载)
    // 要求: 16kHz, 单声道, float32 [-1.0, 1.0] 或 int16
    std::vector<float> audio;  // 用户加载音频到此
    result = engine->Recognize(audio);

    return 0;
}
```

### 流式回调示例

```cpp
#include "asr_service.h"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace SpacemiT;

// 自定义回调
class MyCallback : public AsrEngineCallback {
public:
    void OnOpen() override {
        std::cout << "开始识别" << std::endl;
    }

    void OnEvent(std::shared_ptr<RecognitionResult> result) override {
        if (result->IsSentenceEnd()) {
            std::cout << "最终: " << result->GetText() << std::endl;
        } else {
            std::cout << "中间: " << result->GetText() << "\r" << std::flush;
        }
    }

    void OnComplete() override {
        std::cout << "识别完成" << std::endl;
    }

    void OnError(std::shared_ptr<RecognitionResult> result) override {
        std::cerr << "错误: " << result->GetText() << std::endl;
    }

    void OnClose() override {
        std::cout << "会话关闭" << std::endl;
    }
};

int main() {
    // 创建引擎
    AsrConfig config = AsrConfig::Preset("sensevoice");
    auto engine = std::make_shared<AsrEngine>(config);

    // 设置回调并启动
    engine->SetCallback(std::make_shared<MyCallback>());
    engine->Start();

    // 模拟发送音频 (实际使用时从麦克风/文件读取)
    // 音频格式: 16kHz, 16bit, mono, PCM
    std::vector<uint8_t> audio_chunk(3200);  // 100ms 音频
    for (int i = 0; i < 30; i++) {  // 发送 3 秒
        // audio_chunk = 从麦克风读取...
        engine->SendAudioFrame(audio_chunk);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 停止识别
    engine->Stop();

    return 0;
}
```

### 用户 VAD + flush() 示例

使用 `Flush()` 配合用户自己的 VAD 实现真正的边录边识别：

```cpp
#include "asr_service.h"
#include <iostream>

using namespace SpacemiT;

// 简单的能量 VAD (示例)
class SimpleVAD {
public:
    void feed(const std::vector<int16_t>& audio) {
        // 计算音频能量
        float energy = 0;
        for (int16_t s : audio) {
            energy += s * s;
        }
        energy /= audio.size();

        if (energy < silence_threshold_) {
            silence_frames_++;
        } else {
            silence_frames_ = 0;
        }
    }

    bool is_sentence_end() const {
        return silence_frames_ >= min_silence_frames_;
    }

    void reset() { silence_frames_ = 0; }

private:
    float silence_threshold_ = 100.0f;
    int silence_frames_ = 0;
    int min_silence_frames_ = 10;  // 10 帧静音 = 句子结束
};

int main() {
    AsrConfig config = AsrConfig::Preset("sensevoice");
    auto engine = std::make_shared<AsrEngine>(config);
    engine->SetCallback(std::make_shared<MyCallback>());
    engine->Start();

    SimpleVAD vad;

    // 模拟音频流
    while (has_audio()) {
        auto audio = read_audio_frame();  // 获取音频帧
        engine->SendAudioFrame(audio);

        vad.feed(audio);
        if (vad.is_sentence_end()) {
            engine->Flush();  // VAD 检测到句子结束，立即识别
            vad.reset();
        }
    }

    engine->Stop();
    return 0;
}
```


---

## Python API

```python
"""
Space ASR Python API
"""
import spacemit_asr
from spacemit_asr import Engine, Config, Language, Result

# =============================================================================
# 快捷函数
# =============================================================================

def recognize_file(file_path: str, model_dir: str = "~/.cache/models/asr/sensevoice") -> str:
    """
    识别音频文件 (一行代码)

    Args:
        file_path: 音频文件路径
        model_dir: 模型目录

    Returns:
        识别文本
    """

def recognize_audio(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """
    识别音频数组

    Args:
        audio: numpy 数组 (float32 或 int16)
        sample_rate: 采样率

    Returns:
        识别文本
    """

# =============================================================================
# Language - 支持的语言
# =============================================================================

class Language(Enum):
    AUTO = ...  # 自动检测
    ZH = ...    # 中文
    EN = ...    # 英文
    JA = ...    # 日语
    KO = ...    # 韩语
    YUE = ...   # 粤语

# =============================================================================
# Config - 配置
# =============================================================================

class Config:
    """ASR 配置"""

    def __init__(self, model_dir: str = "~/.cache/models/asr/sensevoice"):
        """
        Args:
            model_dir: 模型目录路径
        """

    @property
    def language(self) -> Language:
        """识别语言"""

    @language.setter
    def language(self, value: Language): ...

    @property
    def sample_rate(self) -> int:
        """采样率 (Hz)"""

    @property
    def punctuation_enabled(self) -> bool:
        """是否自动添加标点"""

    @punctuation_enabled.setter
    def punctuation_enabled(self, value: bool): ...

    # 链式配置
    def with_language(self, language: Language) -> "Config": ...
    def with_punctuation(self, enabled: bool) -> "Config": ...

# =============================================================================
# Result - 识别结果
# =============================================================================

class Result:
    """识别结果"""

    @property
    def text(self) -> str:
        """完整识别文本"""

    @property
    def sentences(self) -> list:
        """句子列表"""

    @property
    def audio_duration_ms(self) -> int:
        """音频时长 (毫秒)"""

    @property
    def processing_time_ms(self) -> int:
        """处理时间 (毫秒)"""

    @property
    def rtf(self) -> float:
        """实时率 (处理时间 / 音频时长)"""

    @property
    def is_empty(self) -> bool:
        """是否为空"""

    def __str__(self) -> str: ...
    def __bool__(self) -> bool: ...

# =============================================================================
# Engine - 识别引擎
# =============================================================================

class Engine:
    """
    ASR 引擎

    推荐使用 context manager:
        with Engine() as engine:
            result = engine.recognize_file("audio.wav")
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Args:
            config: ASR 配置，不传则使用默认配置
        """

    def initialize(self, config: Optional[Config] = None) -> "Engine":
        """
        初始化引擎

        Returns:
            self (支持链式调用)
        """

    def shutdown(self):
        """关闭引擎并释放资源"""

    @property
    def is_initialized(self) -> bool:
        """是否已初始化"""

    def recognize(self, audio: np.ndarray) -> Result:
        """
        识别音频数据

        Args:
            audio: numpy 数组 (float32 或 int16, 16kHz, mono)

        Returns:
            识别结果
        """

    def recognize_file(self, file_path: str) -> Result:
        """
        识别音频文件

        Args:
            file_path: 音频文件路径 (WAV, MP3 等)

        Returns:
            识别结果
        """

    def set_language(self, language: Language):
        """设置识别语言"""

    def set_punctuation(self, enabled: bool):
        """设置是否自动添加标点"""

    # 流式识别 API
    def start(self, callback: AsrCallback = None):
        """启动流式识别会话"""

    def send_audio_frame(self, buffer: bytes):
        """发送音频帧 (PCM S16LE, 16kHz, mono)"""

    def flush(self):
        """立即识别当前缓冲区 (不关闭会话)"""

    def stop(self):
        """停止流式识别会话"""

    @property
    def is_streaming(self) -> bool:
        """是否正在流式识别"""

    # Context Manager
    def __enter__(self) -> "Engine": ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
```

### Python 示例

```python
import spacemit_asr
import numpy as np

# 快捷方式
text = spacemit_asr.recognize_file("audio.wav")
print(text)

# Engine 类
config = spacemit_asr.Config("~/.cache/models/asr/sensevoice")
config.language = spacemit_asr.Language.ZH
config.punctuation_enabled = True

with spacemit_asr.Engine(config) as engine:
    # 文件识别
    result = engine.recognize_file("audio.wav")
    print(f"文本: {result.text}")
    print(f"RTF: {result.rtf:.3f}")

    # 内存识别
    audio = np.zeros(16000, dtype=np.float32)  # 1 秒静音
    result = engine.recognize(audio)

# 链式配置
config = spacemit_asr.Config().with_language(spacemit_asr.Language.EN).with_punctuation(False)
```

---

## 流式识别

Python 流式识别使用回调模式，边录边识别。

### AsrCallback

```python
from spacemit_asr import AsrCallback

class MyCallback(AsrCallback):
    """自定义回调处理类"""

    def on_open(self) -> None:
        """识别会话开始"""
        print("开始识别")

    def on_event(self, result) -> None:
        """
        收到识别结果
        - result.sentences[-1].is_final = False: 中间结果
        - result.sentences[-1].is_final = True: 最终结果
        """
        text = result.text
        is_final = result.sentences[-1].is_final if result.sentences else True

        if is_final:
            print(f"最终: {text}")
        else:
            print(f"中间: {text}", end='\r')

    def on_complete(self) -> None:
        """识别完成"""
        print("识别完成")

    def on_error(self, result) -> None:
        """发生错误"""
        print(f"错误: {result.message}")

    def on_close(self) -> None:
        """会话关闭"""
        print("会话关闭")
```

### 流式识别示例

```python
import time
import spacemit_asr
from spacemit_asr import AsrCallback
import spacemit_audio
from spacemit_audio import AudioCapture

class StreamingCallback(AsrCallback):
    def on_event(self, result):
        print(f">>> {result.text}")

# 配置音频采集 (16kHz, mono)
spacemit_audio.init(sample_rate=16000, channels=1, chunk_size=3200)

# 创建回调和引擎
callback = StreamingCallback()

with spacemit_asr.Engine() as engine:
    # 启动流式识别
    engine.start(callback=callback)

    # 音频回调 - 将数据发送给 ASR
    def on_audio(data: bytes):
        engine.send_audio_frame(data)

    # 启动音频采集
    with AudioCapture() as cap:
        cap.set_callback(on_audio)
        cap.start()
        time.sleep(10)  # 录制 10 秒

    # 停止识别
    engine.stop()
```

### 用户 VAD + flush() 示例

使用 `flush()` 配合用户自己的 VAD 实现真正的边录边识别：

```python
import spacemit_asr
from spacemit_asr import AsrCallback

class SimpleVAD:
    """简单的能量 VAD (示例)"""
    def __init__(self, threshold=500, min_silence_frames=10):
        self.threshold = threshold
        self.min_silence_frames = min_silence_frames
        self.silence_frames = 0
        self.buffer = []

    def feed(self, audio_bytes):
        import numpy as np
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        self.buffer.extend(audio.tolist())

        # 计算能量
        energy = np.mean(audio.astype(np.float32) ** 2)
        if energy < self.threshold:
            self.silence_frames += 1
        else:
            self.silence_frames = 0

    def is_sentence_end(self):
        return self.silence_frames >= self.min_silence_frames

    def reset(self):
        self.silence_frames = 0
        self.buffer = []

class MyCallback(AsrCallback):
    def on_event(self, result):
        print(f"[识别结果]: {result.text}")

# 使用示例
vad = SimpleVAD()
callback = MyCallback()

with spacemit_asr.Engine() as engine:
    engine.start(callback=callback)

    while has_audio():
        audio_chunk = read_audio_frame()  # 获取音频帧
        engine.send_audio_frame(audio_chunk)

        vad.feed(audio_chunk)
        if vad.is_sentence_end():
            engine.flush()  # VAD 检测到句子结束，立即识别
            vad.reset()

    engine.stop()
```

### 依赖

```bash
# 编译 audio 组件
cd ../audio && mkdir -p build && cd build
cmake .. && make -j$(nproc)

# 设置路径
export PYTHONPATH=$PYTHONPATH:/path/to/audio/python
```

---

## 数据格式

- **采样率**: 16000 Hz
- **声道**: 单声道 (mono)
- **格式**:
  - C++: `std::vector<float>` 或 `std::vector<int16_t>`
  - Python: `np.ndarray` (float32 或 int16)
- **范围**:
  - float32: [-1.0, 1.0]
  - int16: [-32768, 32767]

```python
# PCM16 bytes -> float32
audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

# float32 -> PCM16 bytes
pcm_bytes = (audio * 32767).astype(np.int16).tobytes()
```
