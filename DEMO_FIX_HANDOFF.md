# dice_demo 修复交接（方案 A：四处补丁，止住「常驻僵尸化」）

> 交接对象：另一个 AI 会话。仓库：`/home/heweijie/spacemit-k3-dev/projects/dice-game/dice_demo`（分支 `hwj_dev`，工作区干净）。
> 背景：主项目（`../main`，dice-game）通过 robot_arm_nero provider 以 JSONL 常驻方式驱动本 demo（`bash run.sh control --execute`）。2026-09-25 晚实测发现 SDK worker 子进程会静默死亡，死亡后常驻进程**活着但对所有动作命令瞬时秒拒**，直到下一条需要复活的命令才重拉进程（8-13 秒）。这是游戏侧「机械臂不动 / 常驻疑似被重启」的直接根因。
> **目标不变量：常驻进程必须永远处于「能执行动作，或明确失败并自愈」的状态；绝不允许「活着但所有动作秒拒」的僵尸态。**

## 一、证据（可直接查证）

死亡链（三步，全部实锤）：
1. SDK worker（`green_sdk_worker.py` 子进程）静默死亡——0 字节 `sdk_worker.log`、dmesg 无 OOM/段错误、can0 零 bus-off。死因待定，今晚两次活体复现：
   - `datasets/game_20260925_200438_2013248c/runs/`：boot 归位 200443 回执有效，200508 起 patrol 探针目录全空（**空闲期死亡**）；
   - `datasets/game_20260925_202107_98655d8e/runs/`：探针 203839/203842 回执有效，203842 的出拳预备 action 目录为空（动作第一步即死），203923 起全部失败（**动作间隙死亡**，~300ms 窗口）。
2. 父进程管道被关：`green_runtime.py` `_call_once` 的 `except BaseException: self._terminate()` 在**一切**失败（包括 worker 正常上报的动作失败回执）时把管道杀掉。
3. 错误被误吞：`green_control.py` `_run_action` 的 `except ValueError` 分支把「写死管道」抛出的 `ValueError: I/O operation on closed file` 当成「未知动作名」→ `rejected(unknown_action)` → **既不触发恢复、也不退出** → 僵尸。主侧 web 日志铁证一行：`[robot] post-round ensure_home: rejected(unknown_action): I/O operation on closed file.`

同一链条还有更早的形态（同病根）：傍晚 4 个会话（`171527`/`172905`/`172926`/`173014`）state.json 的 error 均为 `恢复失败（home）: ValueError: I/O operation on closed file.`——失败后 `_recover` 不重启 worker，归位写在死管道上必炸。**这四处补丁同时覆盖今晚和傍晚两类崩溃。**

## 二、修复清单（按优先级）

### P0-1 错误归类：通道 I/O 错误绝不能算 unknown_action

位置：`cup_grasp_demo/flow/green_runtime.py` `_call_once`（约 140-170 行）；配合 `green_control.py:240-246` 的 `except ValueError` 分支。

根因：worker 死后父进程管道被 `_terminate()` 关闭，下一次 `client.call()` 在 `self.process.stdin.write(...)` 处抛 `ValueError: I/O operation on closed file`，一路传到 `_run_action` 的 ValueError 分支（本意只接 `registry.recipe()` 的「未知动作名」）。

改法（推荐在 green_runtime 统一归类，一处改动全链路生效）——`_call_once` 里包住 stdin 写入：

```python
        try:
            message = json.dumps(message) + '\n'   # 现有代码
            try:
                self.process.stdin.write(message)
                self.process.stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as exc:
                # 管道已断（worker 死亡，或父进程此前已 close）：
                # 统一归类为 worker 死亡，交给 call() 的 worker-exited
                # 分支和 _recover 的重启逻辑接管，绝不让它以裸 ValueError
                # 漏出去被误判成「未知动作名」。
                raise RuntimeError(
                    f'SDK worker exited (write failed): {type(exc).__name__}: {exc}'
                ) from exc
```

验收：worker 死后再发 action，事件流应是 `failed`（随后走恢复），而不是 `rejected(unknown_action): I/O operation on closed file`。

### P0-2 僵尸总闸：`_recover` 归位前重启 SDK worker

位置：`cup_grasp_demo/flow/green_control.py:102-105`（现在的条件）：

```python
        if "SDK worker exited" in str(error_text):
            sdk = getattr(self.flow, "_sdk", None)
            if sdk is not None:
                sdk.restart()
```

根因：错误文本几乎永远不含 `"SDK worker exited"`（那个字符串只在 worker **中途 EOF** 时产生；固件失联、限位超时、管道已关都不含），所以 `restart()` 永远不执行，归位写在死管道上。

改法：**无条件重启**（删掉字符串匹配）：

```python
        try:
            # 归位前统一重拉 SDK worker：worker 死亡是失败的常见形态，
            # 不重启则归位必然写在死管道上。restart() 自身保证旧进程先收掉。
            sdk = getattr(self.flow, "_sdk", None)
            if sdk is not None:
                sdk.restart()
            self._reply("recovery_started", request_id, action="home", ...)
```

配套（必须一起改）：`green_runtime.py:56-64` 的 `restart()` 目前假设「旧进程已死、只关管道」——对**活着的** worker 调用会泄漏进程（它持有 `/tmp/nero_can0_control.lock` 和 CAN 通道，新 worker 会启动失败）：

```python
    def restart(self):
        """Replace a possibly-dead worker; a still-alive process is terminated first."""
        if self.process.poll() is None:
            self._terminate()   # 活着的先收掉：防泄漏、防 CAN 控制锁被占
        for pipe in (self.process.stdin, self.process.stdout):
            ...  # 现有代码
```

### P1-3 别误杀活 worker：回执失败 ≠ worker 死亡

位置：`green_runtime.py` 两处——`_call_once` 末尾的 `except BaseException: self._terminate(); raise`（约 166-170），以及 `call()` 的 `if 'SDK worker exited' not in str(exc): self.close(); raise`（约 112-114）。

根因：worker 上报「动作失败」（receipt success=false，如固件 None、Hand start、J7 限位超时）时它自己**活着且空闲**，此刻杀管道等于把一切可恢复失败变成死管道。

改法：
- `_call_once`：把「回执失败」的 `raise RuntimeError(f"{report.get('error')}; ...")` 挪到 try/except **之外**（或在 except 里豁免这一类），使其不触发 `_terminate()`。超时（receive 的 TimeoutError，worker 可能卡死）和 EOF 保留 terminate——那是真需要收线的情况。
- `call()`：非 worker-exit 的错误**只 raise 不 close()**，保住连接给恢复链复用；worker-exited 分支（回执打捞、零传输重启）保持不变。

### P1-4 worker 死因插桩：让下一次死亡可归因

位置：`cup_grasp_demo/flow/green_sdk_worker.py` 的两个**干净出口**（都不留任何日志）：

```python
                if message.get('command') == 'close':
                    print('[worker] exit: close command', file=sys.stderr, flush=True)
                    break
                ...
                if code and not (message['command'] == 'shake' and retryable_shake_start('shake', report)):
                    print(f'[worker] exit: failed motion code={code} output={output}',
                          file=sys.stderr, flush=True)
                    break
```

循环自然结束（stdin EOF）后在 finally 里补一行 `[worker] exit: eof`。stderr 由父进程路由到 session 目录的 `sdk_worker.log`。信号死亡本身有 traceback 可见；0 字节 = 干净出口，插桩后即可区分。

> 注：worker 两次死亡（空闲 ~25s 后、动作间隙 ~300ms 窗口）**根因未定**，插桩是定位手段。若查实是「空闲死亡」类，可再评估 worker 内加心跳/保活——先不要做，等插桩数据。

## 三、不要改的东西（与主侧的契约）

- JSONL 事件形状与 id 关联：`ready / phase_started / command_completed / action_completed / actions_reloaded / rejected / failed / recovery_started / recovered / pose / run_completed / closed`。主侧 provider 严格按 id 关联判完成。
- 失败自愈语义：`failed` → `recovery_started` → 归位 → `recovered`（进程不退、继续服务）；恢复自身失败发 `failed(phase="recovery:home")` 并走 legacy exit。主侧（commit c66698a）已按此适配。
- `query_pose` 的拒绝语义：执行器忙回 `command_busy`、探针失败回 `rejected(pose_unavailable)`——保持现状（主侧巡逻只认明确的 at_home=false）。
- `reload` 语义：fire-and-forget，失败保留旧表。主侧在每个 action 前发它。
- 恢复成功后 serve() 必须继续服务（`_handle` 返回 True）——不能退回「失败即退出」的旧语义。

## 四、验证

**板端手工（核心场景，用 `scripts/control_console.py` 真机模式）**：
```bash
# 模拟 worker 死亡
kill $(pgrep -f green_sdk_worker.py)
# console 里发一条手势命令。预期（当前是坏的）：
#   坏现状：rejected(unknown_action): I/O operation on closed file，之后所有命令一直秒拒
#   修好后：failed → recovery_started → recovered（重拉 worker 并归位）→ 后续命令正常
```
另一个场景（worker 活着、动作回执失败，如固件失联）：预期恢复归位走**同一条连接**成功（不再 `恢复失败（home）: I/O ...`）。

**单测**（按仓库现有测试风格补，场景清单）：
1. 管道写失败（先 `runtime.close()` 再 `call`）→ 异常信息含 `SDK worker exited`，`_run_action` 走 failed/恢复而非 unknown_action 拒绝；
2. 动作回执失败（worker 活着）→ worker 进程未被杀、`_recover` 归位成功；
3. `_recover` 无条件 restart：死 worker 归位成功；活 worker 场景旧进程被收掉（无泄漏，`/tmp/nero_can0_control.lock` 可立即重入）；
4. worker 三个干净出口各留一行 stderr；
5. console 回归：failed → recovery 序列、actions_reloaded 契约不回退。

## 五、范围纪律

只做这四处，**不顺手重构**。今晚的游戏侧问题与此正交（主侧取消分级已单独提交 285d0ba）。每处补丁独立可回退，改完跑一遍板端手工验证即可交给游戏侧复测。
