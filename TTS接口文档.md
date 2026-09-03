# GPT-SoVITS v2ProPlus 流式 TTS 接口文档

> 服务版本:v2ProPlus(fp16) | 更新日期:2026-09-01
> 完整部署说明见 `./TTS部署说明.md`
> 服务启停:`bash ./start.sh`(一键拉起,自动预热 zh/en/ja)/ `bash ./stop.sh`(一键停止,
> 带看门狗抑制);`bash ./selftest.sh`(13 项端点回归自检);
> 内置保活心跳(每 10 分钟一次微小合成),任意时刻请求都是热态首包

---

## 1. 服务概览

| 服务 | 地址(Tailscale) | 说明 |
|---|---|---|
| **音色管理后台 + 按名调用 ⭐推荐** | `http://<服务IP>:9873` | 设备接这个:`/tts` 克隆/专属双模式;`/ui` 管理界面(顶部全局模式开关 + 双模式独立工作区) |
| GPT-SoVITS api_v2 直连 | `http://<服务IP>:9880` | 每次请求带参考音频路径(下文完整文档) |
| 流式测试 WebUI | `http://<服务IP>:9872` | 网页试听测试 |

| 项目 | 值 |
|---|---|
| 鉴权 | 无(仅限 Tailscale 网络内使用) |
| 音频输出格式 | 32000 Hz / 16bit / 单声道 / little-endian |
| 流式支持 | ✅ HTTP chunked 分块传输(边合成边返回) |
| 语言 | 官方 v2 全部 **11 种语言模式**(zh/ja/en/ko/yue/auto/auto_yue/all_zh/all_ja/all_yue/all_ko),管理后台已全量暴露,ko/yue/auto/all_zh 实测通过 |
| 并发能力 | 单实例串行(同一时间处理一个请求,多请求自动排队) |
| 保活 | start.sh 内置三语预热 + 10 分钟心跳,闲置后首包无惩罚 |

**端点一览**

| 端点 | 方法 | 用途 |
|---|---|---|
| **9873** `/tts` | POST | **按音色名合成(推荐)**,兼容透传 9880 请求体 |
| **9873** `/voices` | GET / POST / DELETE | 音色注册表(列表 / 注册 / 删除) |
| **9873** `/voices/{id}` | PATCH | 修改音色(改ID重命名 / 转写 / 语言 / 备注) |
| **9873** `/voices/{id}/audio` | POST | 替换音色的参考音频(multipart,re_asr=true 自动重识别转写) |
| **9873** `/models` 系列 | GET / POST / DELETE / POST activate | 微调模型注册表与热切换(见「微调模型管理」) |
| **9873** `/models/upload` | POST(multipart) | **上传专属音色包**注册(gpt_file + sovits_file + ref_file + id, 转写可自动ASR) |
| **9873** `/voices/{id}` PATCH | 可含 `model_id` | **音色绑定微调模型**:调用该音色自动切换模型,未绑定自动用 base |
| **9873** `/ui` | GET | 管理界面(**流式试音**(试音完成后可一键下载本次合成 WAV)、注册/编辑/试听/删除音色、微调模型、备份恢复) |
| **9873** `/asr` | POST | 参考音频自动转写(SenseVoiceSmall,zh/en/ja/ko/yue 自动检测语言) |
| **9873** `/v1/audio/speech` | POST | **OpenAI TTS 兼容端点**(现成客户端即插即用,mp3/wav/flac/opus/aac/pcm) |
| **9873** `/voices/backup` · `/voices/restore` | GET / POST | 音色库一键备份(zip)/ 恢复 |
| 9880 `/tts` | POST(推荐)/ GET | 语音合成(直连,完整参数) |
| 9880 `/control` | GET / POST | 服务控制(restart/exit) |
| 9880 `/set_gpt_weights` | GET | 热切换 GPT(t2s)模型 |
| 9880 `/set_sovits_weights` | GET | 热切换 SoVITS 模型 |

---

## 2. 快速开始

### 方式一:克隆模式(按音色名调用,推荐,9873)

音色先在管理界面 http://<服务IP>:9873/ui 注册(上传音频+填转写),之后任何设备:

```bash
curl -X POST http://<服务IP>:9873/tts \
  -H "Content-Type: application/json" \
  -d '{"voice": "demo_female_zh", "text": "你好,这是一段测试语音。"}' \
  -o out.wav
```

- `voice` 缺省时用注册表 `settings.default_voice`(若设置);`text_lang`/`streaming_mode`/`speed` 缺省为内置值
- 兼容透传:带 `ref_audio_path` 的完整 9880 请求体发到 9873 也照常工作

**按名调用可用参数**(均可选;缺省值为内置常量:streaming_mode=3、speed=1.0、text_lang=zh):

| 参数 | 默认 | 说明 |
|---|---|---|
| `voice` | 注册表 default_voice(若设置) | 注册表中的音色 ID |
| `text` **必填** | — | 要合成的文本 |
| `text_lang` | zh | 11 种语言模式任一 |
| `streaming_mode` | 3 | 整数 2/3 才是真流式 |
| `speed` | 1.0 | 语速 0.5~2.0 |
| `media_type` | wav | wav / raw / ogg / aac |
| `min_chunk_length` | 16 | 流式切块粒度(勿动) |
| `top_k` / `top_p` / `temperature` | 15 / 1.0 / 1.0 | 采样参数:越小越稳,越大越有情感 |
| `repetition_penalty` | 1.35 | 出现复读/卡字就调大(1.0~3.0) |
| `seed` | -1(随机) | 固定数值可复现同一结果 |
| `text_split_method` | cut5 | 长文本切句(cut0~cut5) |
| `fragment_interval` | 0.3 | 句间静音秒数 |

### 方式二:专属模式(微调模型,9873)

已上传专属音色包(见「微调模型管理」)后,传 `model` 参数即可,自动切换模型并使用其捆绑参考音频:

```bash
curl -X POST http://<服务IP>:9873/tts \
  -H "Content-Type: application/json" \
  -d '{"model": "anke_ft", "text": "你好,这是专属音色。", "streaming_mode": 3}' \
  -o out.wav
```

- `model` 与 `voice` 二选一(同时传时 `model` 优先);其余参数与克隆模式完全一致
- 首次调用会热切换引擎到该模型(数秒,含自动预热),之后为热态

### 方式三:直连 api_v2(9880,完整参数)

```bash
curl -X POST http://<服务IP>:9880/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好,这是一段测试语音。",
    "text_lang": "zh",
    "ref_audio_path": "./voices/demo_female_zh.wav",
    "prompt_text": "希望你以后能够做的比我还好呦。",
    "prompt_lang": "zh",
    "media_type": "wav",
    "streaming_mode": 3
  }' -o out.wav
```

> ⚠️ `streaming_mode` 必须传**整数 2 或 3** 才是真流式。传 `true` 会因
> Python `True==1` 落入旧版"整句合成完再返回"模式。

---

## 3. `/tts` 完整参数

POST JSON body(GET 则为同名 query 参数):

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `text` **必填** | str | — | 要合成的文本 |
| `text_lang` **必填** | str | — | 文本语言,见下方语言表 |
| `ref_audio_path` **必填** | str | — | **服务端本地**参考音频路径(3~10s 干净人声),如 `./voices/xxx.wav` |
| `prompt_lang` **必填** | str | — | 参考音频的语言 |
| `prompt_text` | str | "" | 参考音频的逐字转写。**强烈建议提供**,音色相似度和稳定性明显更好 |
| `aux_ref_audio_paths` | list[str] | [] | 附加参考音频(多音色融合,一般不用) |
| `streaming_mode` | int/bool | 0 | **0=非流式;1/true=旧版分段返回(假流式,勿用);2=真流式·质量优先;3=真流式·首包最快** |
| `min_chunk_length` | int | 16 | 流式切块的语义 token 数(16≈0.64s/块)。实测 16 最优,**调小会显著恶化 RTF** |
| `overlap_length` | int | 2 | 相邻块重叠 token 数(消块边界伪影),勿动 |
| `media_type` | str | wav | `wav`=44字节头+裸PCM(推荐流式)/ `raw`=纯 PCM 无头 / `ogg` / `aac` |
| `speed_factor` | float | 1.0 | 语速 0.5~2.0(变速不变调) |
| `text_split_method` | str | cut5 | 长文本切句方式,见下方切分表 |
| `batch_size` | int | 1 | 非流式批处理句数 |
| `top_k` / `top_p` / `temperature` | — | 15 / 1.0 / 1.0 | 采样参数,一般不动 |
| `repetition_penalty` | float | 1.35 | 重复惩罚,出现复读可加大 |
| `seed` | int | -1 | 随机种子(-1 随机;固定种子可复现) |
| `fragment_interval` | float | 0.3 | 多句之间插入的静音秒数 |
| `parallel_infer` | bool | true | 并行推理(流式时自动关闭) |
| `split_bucket` | bool | true | 批次分桶(流式时自动关闭) |

### 语言表(`text_lang` / `prompt_lang` 共用)

| 值 | 含义 |
|---|---|
| `zh` | **中英混合**识别(中文文本夹英文单词用这个) |
| `ja` | **日英混合**识别 |
| `en` | 全部按英文 |
| `ko` | 韩英混合 |
| `yue` | 粤英混合 |
| `auto` | 多语种自动检测切分 |
| `auto_yue` | 多语种自动(中文段按粤语) |
| `all_zh` / `all_ja` / `all_ko` / `all_yue` | 强制整段按单一语种(含汉字)识别 |

### 长文本切分表(`text_split_method`)

| 值 | 含义 | 适用 |
|---|---|---|
| `cut0` | 不切,整句直接进模型 | **短句/对话式实时合成(配合流式最可控)** |
| `cut5` | 按标点切 | **长段文本默认推荐** |
| `cut3` | 按中文句号切 | 中文长文 |
| `cut4` | 按英文句号切 | 英文长文 |
| `cut1` | 凑四句一切 | 整段朗读 |
| `cut2` | 凑 50 字一切 | 长段落 |

> 流式模式下切句影响块的自然性:`cut0` 适合一句话的请求;几百字长文用 `cut5`。

---

## 4. 流式响应格式(客户端接入重点)

### 4.1 线上格式(`media_type=wav`)

```
HTTP/1.1 200 OK
Content-Type: audio/wav            ← 响应头无 Content-Length(chunked)

┌────────────────────────────────────────────────────┐
│ 字节 0~43   : 标准 WAV 头(44 字节,RIFF/data)       │ ← 只出现一次
│ 字节 44~... : 裸 PCM int16 LE 单声道 32000Hz        │ ← 连续不断
└────────────────────────────────────────────────────┘
```

- 每个 HTTP chunk 到达即可处理,**不用等响应结束**
- 后续块**不再重复 WAV 头**(跳过 44 字节后全部是音频)
- 客户端读满 44 字节后应解析 `buf[24:28]`(采样率)并立即开始播放

### 4.2 `media_type=raw`(嵌入式推荐)

响应从第 0 字节起就是纯 PCM(int16 LE / 单声道 / 32000Hz),没有任何封装。
客户端需自行按 32000Hz 播放。格式最简单,适合 ESP32 等直接喂 I2S/DAC。

### 4.3 `media_type=ogg / aac`

每个块单独编码为 ogg/aac 流,可直接喂支持流式的播放器(如 ffplay、部分浏览器),
但**编码引入额外延迟**,追求实时首包建议用 wav/raw。

### 4.4 读取节奏建议

- **不要**按固定字节数等待——应"到多少处理多少"(HTTP chunked 推送)
- 服务端每块约 0.64s 音频(约 40KB);RTF≈0.15,播放消耗远慢于生成,不会断流
- 若自己做缓冲,建议 **0.3~0.6s 起播水位**,过大反而增加感知延迟

---

## 5. 各语言客户端示例

### 5.1 curl

```bash
# 流式(整段落盘,边下边写)
curl -N -X POST http://<服务IP>:9880/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"流式测试。","text_lang":"zh","ref_audio_path":"./voices/demo_female_zh.wav","prompt_text":"希望你以后能够做的比我还好呦。","prompt_lang":"zh","streaming_mode":3}' \
  -o stream.wav

# 播放流(需要声音直接出)
curl -N -X POST http://<服务IP>:9880/tts -H "Content-Type: application/json" \
  -d '{...同上, "media_type":"raw"...}' | ffplay -f s16le -ar 32000 -ch_layout mono -i -
```

### 5.2 Python(requests,边收边播,免等待)

```python
import io, wave, requests, numpy as np, sounddevice as sd

API = "http://<服务IP>:9880/tts"
payload = {
    "text": "你好,来自 Python 的流式调用。",
    "text_lang": "zh",
    "ref_audio_path": "./voices/demo_female_zh.wav",
    "prompt_text": "希望你以后能够做的比我还好呦。",
    "prompt_lang": "zh",
    "media_type": "wav",
    "streaming_mode": 3,          # 整数!
}

r = requests.post(API, json=payload, stream=True, timeout=120)
it = r.iter_content(chunk_size=65536)
buf, stream, sr = b"", None, 32000
for chunk in it:
    if stream is None:                      # 先收满 44 字节头
        buf += chunk
        if len(buf) < 44:
            continue
        sr = int.from_bytes(buf[24:28], "little")
        stream = sd.OutputStream(samplerate=sr, channels=1, dtype="int16")
        stream.start()
        audio = np.frombuffer(buf[44:], dtype=np.int16)
    else:
        audio = np.frombuffer(chunk, dtype=np.int16)
    stream.write(audio)                     # 立刻播放(内部自动背压)
stream.stop()
```

现成脚本:`./bench/stream_play.py`(支持 `--save`、无声卡降级)。

### 5.3 浏览器 JavaScript(Web Audio,网页即点即播)

```javascript
const resp = await fetch("http://<服务IP>:9880/tts", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    text: "你好,浏览器流式播放测试。",
    text_lang: "zh",
    ref_audio_path: "./voices/demo_female_zh.wav",
    prompt_text: "希望你以后能够做的比我还好呦。",
    prompt_lang: "zh", media_type: "wav", streaming_mode: 3,
  }),
});
const reader = resp.body.getReader();
const ctx = new AudioContext({ sampleRate: 32000 });
let buf = new Uint8Array(0), nextAt = 0, header = 44;
const play = (pcm) => {                       // pcm: Int16 单声道
  const f = ctx.createBuffer(1, pcm.length, 32000).getChannelData(0);
  for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 32768;
  const src = ctx.createBufferSource(); src.buffer = ctx.createBuffer(1, pcm.length, 32000);
  src.buffer.getChannelData(0).set(f);
  nextAt = Math.max(nextAt, ctx.currentTime + 0.05);
  src.start(nextAt); nextAt += pcm.length / 32000;
};
for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  const merged = new Uint8Array(buf.length + value.length);
  merged.set(buf); merged.set(value, buf.length); buf = merged;
  const usable = buf.length - header - ((buf.length - header) % 2);
  if (usable <= 0) continue;
  const bytes = buf.subarray(header, header + usable);   // 首块 header=44, 之后为 0
  play(new Int16Array(bytes.buffer, bytes.byteOffset, usable / 2));
  buf = buf.slice(header + usable); header = 0;
}
```

> 跨域说明:若网页与 API 不同源,浏览器会拦 CORS。最省事的方案是用现成的
> 流式 WebUI(9872 端口,同源封装)或把页面挂在与 API 同源的反代下。
> 注意 `AudioContext` 构造时显式给 `sampleRate: 32000`,否则部分浏览器按
> 默认 48k 解码会导致变速变调。

### 5.4 嵌入式设备(ESP32 等)要点

- 请求:`POST /tts`,`media_type: "raw"`(免解析 WAV 头)
- 收到的字节流即 **int16 LE 单声道 32000Hz PCM**,按序写入 I2S/DAC
- 关键:下载速度(RTF 0.15,即 1s 音频 0.15s 传完)远快于播放,**必须做环形缓冲**
  起播水位约 0.3~0.5s(约 20~32KB),缓冲区建议 ≥2s(128KB)防抖动
- 换音色:参考音频须先放到服务端 `./voices/`,设备只传路径

---

## 6. 错误处理

| HTTP 码 | 场景 | 响应 |
|---|---|---|
| 200 | 成功 | 音频流(audio/wav 等) |
| 400 | 缺少必填字段(如 `"message":"ref_audio_path is required"`) | JSON `{"message": "..."}` |
| 400 | 参数不合法(语言/媒体类型/streaming_mode 取值等) | JSON `{"message": "..."}` |
| 400 | 参考音频路径不存在/格式错误 | JSON `{"message": "..."}` |

(实测所有错误统一为 400 + `{"message": ...}`;客户端解析 `message` 字段即可。)

客户端注意:失败时没有音频流,应先检查 `status_code==200` 再开始播放;
流中途断开(`Response ended prematurely`)通常是服务端异常,查 `./api_v2.log`。

---

## 7. 其他端点

### `/control` — 服务控制

```bash
curl "http://<服务IP>:9880/control?command=restart"   # 重启(重新加载模型, 约20s)
curl "http://<服务IP>:9880/control?command=exit"      # 退出进程
```

### `/set_gpt_weights`、`/set_sovits_weights` — 热切换模型(微调后用)

```bash
curl "http://<服务IP>:9880/set_gpt_weights?weights_path=GPT_SoVITS/logs/xxx/GPT-even.pth"
curl "http://<服务IP>:9880/set_sovits_weights?weights_path=GPT_SoVITS/logs/xxx/SoVITS-eighth.pth"
```

切换立即生效、不用重启服务,所有后续请求用新音色模型。

### 9873 音色注册表 API(程序化管理音色)

```bash
# 列出全部音色
curl http://<服务IP>:9873/voices

# 注册音色(音频须已在服务端磁盘上;网页上传请用管理界面 /ui)
curl -X POST http://<服务IP>:9873/voices \
  -H "Content-Type: application/json" \
  -d '{"voice_id":"xiaoming","file_path":"./voices/xiaoming.wav",
       "prompt_text":"参考音频的转写","prompt_lang":"zh","note":"产品音色"}'

# 删除音色(注册表条目删除,磁盘文件保留)
curl -X DELETE http://<服务IP>:9873/voices/xiaoming

# 修改音色: 改ID(重命名,音频文件同步改名)/ 转写 / 语言 / 备注(字段可选)
curl -X PATCH http://<服务IP>:9873/voices/xiaoming \
  -H "Content-Type: application/json" \
  -d '{"voice_id":"xiaoming_new","prompt_text":"修正后的转写","note":"产品音色"}'
```

### 9873 `/asr` — 参考音频自动转写

```bash
curl -X POST http://<服务IP>:9873/asr \
  -H "Content-Type: application/json" \
  -d '{"file_path":"./voices/xxx.wav"}'
# → {"text":"识别出的台词","prompt_lang":"zh"}   (语言自动检测)
```
管理界面「克隆模式 → ➕ 添加克隆音色」页有同名按钮:上传音频后点【🎙️ 自动识别转写】即可自动填好
转写文本和参考音频语言(首次使用自动下载 SenseVoiceSmall 模型,约 1GB 显存)。

### 9873 微调模型管理(专属音色热切换)

官方 webui 训练产物是**一对权重**(`GPT_SoVITS/logs/<实验名>/` 下的 `GPT-*.pth` 与 `SoVITS-*.pth`)。
注册后一键启用/切回底模,走官方热切换端点,**无需重启服务**,重启后也会自动恢复启用的模型。

### 专属音色包 = 模型对 + 捆绑参考音频

注册一个微调模型时**同时上传该说话人的参考音频(3~10s)**,转写可自动 ASR 识别——
模型与声音身份捆绑成一个"专属音色包",启用即用,无需再去音色库配对。

**① 网页/接口上传(推荐)**:管理界面「专属模式 → ➕ 添加专属音色(模型包)」上传三个文件即可;接口版:

```bash
curl -X POST http://<服务IP>:9873/models/upload \
  -F "id=anke_ft" -F "note=安可微调" \
  -F "gpt_file=@GPT-anke.pth" -F "sovits_file=@SoVITS-anke.pth" \
  -F "ref_file=@anke_ref.wav" -F "re_asr=true"
```

**② 服务端路径注册(高级)**:

```bash
# 注册模型对(相对路径按 GPT-SoVITS/ 解析, 也可用绝对路径)
curl -X POST http://<服务IP>:9873/models -H "Content-Type: application/json" \
  -d '{"id":"anke_ft","gpt_path":"GPT_SoVITS/logs/anke/GPT-anke.pth",
       "sovits_path":"GPT_SoVITS/logs/anke/SoVITS-anke.pth","note":"安可微调"}'

curl -X POST http://<服务IP>:9873/models/anke_ft/activate   # 启用
curl -X POST http://<服务IP>:9873/models/base/activate      # 切回官方底模
curl http://<服务IP>:9873/models                            # 列表+当前启用状态
curl -X DELETE http://<服务IP>:9873/models/anke_ft          # 删除注册(权重保留)
```

### 音色绑定与自动路由(免手动切换)

给音色绑定微调模型后,**调用时引擎自动切换**——绑定音色自动切到微调模型,未绑定音色自动切回 base:

```bash
curl -X PATCH http://<服务IP>:9873/voices/anke -H "Content-Type: application/json" \
  -d '{"model_id":"anke_ft"}'    # 绑定; 传 "base" 或 "" 解除绑定
curl -X POST http://<服务IP>:9873/tts -d '{"voice":"anke","text":"..."}'   # 自动用 anke_ft
curl -X POST http://<服务IP>:9873/tts -d '{"voice":"其他音色","text":"..."}' # 自动用 base
```

切换含权重加载+自动预热,约数秒;**频繁在绑定/未绑定音色间交替会反复热切换,建议分批使用**。

### 两种模式(自动路由,设备无感)

| 调用方式 | 模式 | 引擎行为 |
|---|---|---|
| `{"voice":"音色ID","text":"..."}` | 克隆模式 | 自动确保 base 底模 + 音色库参考音频 |
| `{"model":"微调模型ID","text":"..."}` | 专属模式 | 自动切换微调模型 + 用其捆绑参考音频(voice 忽略) |
| 音色绑定了模型(PATCH model_id) | 自动 | 调用该音色切绑定模型,其他音色切回 base |

切换含权重加载+自动预热(约数秒),**频繁交替会反复热切换,建议分批使用**。
注意:① 同一时间只有一个模型生效;② 微调模型需基于 v2ProPlus 底模训练;③ 专属模式的
声音身份来自注册时捆绑的参考音频。

### 9873 `/v1/audio/speech` — OpenAI TTS 兼容端点

任何支持 OpenAI TTS 协议的客户端(Home Assistant、各类语音助手框架)无需写对接代码:

```bash
curl -X POST http://<服务IP>:9873/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"要合成的文本","voice":"anke","response_format":"mp3","speed":1.0}' \
  -o out.mp3
```

- `voice` 填注册表音色 ID(克隆模式);也可传 `model` 字段填**微调模型 ID** 进入专属音色模式(自动切模型+用注册时捆绑的参考音频,忽略 voice);填未知值自动回退默认音色
- `response_format`:mp3(默认)/ wav / flac / opus / aac / pcm(mp3 等 Format 走 ffmpeg 转码,秒级)
- `speed` 0.25~4.0;`model` 参数忽略;该端点为整段返回(OpenAI 协议本身不流式),
  需要流式低延迟请用 `/tts`
- OpenAI SDK 用法:`client.audio.speech.create(model="gpt-sovits-v2proplus", voice="anke", input="...")`
  (base_url 设为 `http://<服务IP>:9873/v1`)

### 9873 音色库备份 / 恢复

```bash
# 备份: 打包全部音色音频 + registry.json 下载(zip)
curl http://<服务IP>:9873/voices/backup -o voices_backup.zip

# 恢复: 上传备份包,合并音色(overwrite=true 覆盖同名)
curl -X POST "http://<服务IP>:9873/voices/restore?overwrite=true" -F "file=@voices_backup.zip"
```

管理界面「💾 备份 / 恢复」标签页提供同样的图形化操作。备份包放在
`voices/backups/` 下的会保留;建议定期下载到其他机器存放。

注册表持久化在 `./voices/registry.json`;管理界面里还可设置
默认音色、默认流式模式(2/3)、默认语速、默认文本语言——按名调用缺省时生效。

---

## 8. 换音色(零样本克隆)操作流程

1. 准备 3~10 秒干净人声(无背景音乐/噪音)。**引擎硬性限制:时长必须在 3~10s 内**——上传瞬间即校验,
   超限直接报错拦截(注册时还会再校验一遍);服务器路径注册同样校验;不知道台词?管理界面注册时点
   【🎙️ 自动识别转写】可自动生成转写文本和语言(建议人工核对一遍)
2. **推荐:打开管理界面 http://<服务IP>:9873/ui →「克隆模式 → ➕ 添加克隆音色」**,
   上传音频、填音色ID和转写,点注册即入库
   (命令行方式也可 scp 到 `./voices/` 后 `POST /voices` 注册)
3. 调用时按名引用:
   ```json
   {"voice": "xiaoming", "text": "要合成的话"}
   ```
4. 同一参考音色可用不同 `text_lang` 跨语言合成(音色不变,口音受参考音频影响)

**参考音频质量决定克隆上限**:底噪/混响/背景音乐都会被克隆进去。常用音色想要更高
相似度和稳定性,走 1 分钟数据微调(WebUI 训练)后用 `/set_*_weights` 热加载。

---

## 9. 性能参考(本机 RTX 4060 Laptop 8GB 实测)

| 场景 | 首包延迟 | RTF | 说明 |
|---|---|---|---|
| `streaming_mode=3` | **0.15~0.22s** | 0.16~0.21 | 实时对话/设备端推荐 |
| `streaming_mode=2` | 0.45~0.70s | 0.13~0.15 | 质量优先,停顿处切块 |
| 非流式(mode 0) | 1.4s+ | 0.22~0.42 | 等整段生成完才返回 |

- 服务由 start.sh 启动时自动完成 zh/en/ja 三语预热,并每 10 分钟心跳保活
  (keepalive.sh),**任意时刻请求都是热态首包**,无需关心冷启动
- 播放消耗速度是生成速度的 1/5 以下,正常情况**绝不会出现播着播着断流**
- 单实例串行;第二个并发请求会排队,感知延迟 = 排队 + 生成

---

## 10. FAQ

**Q: 参考音频可以用 HTTP URL 吗?**
不行,`ref_audio_path` 只认服务端本地路径。先传到服务器 `./voices/`。

**Q: 英文/日文也用中文 BERT 吗?**
BERT 特征仅用于中文;en/ja/ko 走各自 G2P,音色克隆不受影响,无需额外模型。

**Q: 为什么我的客户端播放出来全是噪音?**
八成是采样率或位深处理错了:必须是 32000Hz/16bit/单声道/小端。浏览器端
`AudioContext` 必须显式指定 `sampleRate: 32000`。参考 5.2/5.3 示例。

**Q: `streaming_mode: true` 和 `3` 有什么区别?**
`true` 等于 `1`(旧版整句分段返回,合成完才响应);`2/3` 才是边生成边出块的真流式。

**Q: 长文本一次性合成可以吗?**
可以,建议 `text_split_method: "cut5"` + `streaming_mode: 3`,先出的块先播;
非流式整段合成几千字会很久,客户端超时要放宽。

**Q: 图形化测试?**
`bash ./start.sh` 一键拉起全部服务 → 管理界面 http://<服务IP>:9873/ui(流式试音,支持选音色、
进阶采样参数;试音完成后点【⬇️ 下载本次合成 WAV】即可拿走音频文件,导出文件在服务端
`/tmp/tts_exports/` 下保留 24h),测试页 http://<服务IP>:9872

**Q: 服务怎么启动/停止/保活?**
`start.sh` 一键启动(API+测试页+管理后台,自动预热三语);`stop.sh` 一键停止;
心跳脚本 `keepalive.sh` 每 10 分钟自动微小合成保持 GPU 高性能(周期改脚本内 `INTERVAL`),日志 `keepalive.log`。

**Q: 支持哪些语言?**
官方 v2 全部 11 种模式(见第 3 节语言表),管理后台下拉框已全量提供;
ko/yue/auto/all_zh 均实测合成成功。跨语言共用同一音色,直接换 `text_lang` 即可。
