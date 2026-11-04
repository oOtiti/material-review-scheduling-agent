# 素材审核与排期草稿 Agent

第 1 题「素材审核与发布排期 Agent」。读取合成素材和档期数据，审核后提出可确认的时段，等待用户选择，再保存排期草稿。真实模式用 LangChain 选择工具动作。状态机校验审核、时间和用户确认，SQLite 保存跨进程的任务状态与草稿。

Python 3.11 或更新版本。在仓库根目录运行下面的命令。离线模式使用 Python 标准库。

```powershell
python -m unittest discover -s tests -v
python -m src.benchmark --mode offline
$proposal = python -m src.main --db demo.db start examples/normal.input.json | ConvertFrom-Json
$proposal.response
python -m src.main --db demo.db confirm $proposal.request_id $proposal.proposal_id 1
python -m src.main --db demo.db status $proposal.request_id
```

`start` 返回提案和候选时段。检查候选后，将上例中的 `1` 换为你选定的编号。每次演示使用新的数据库文件名可重新走一遍确认流程。相同提案和编号的重复确认返回同一个草稿。

文本入口 `chat` 处理常见问候、能力询问、无关请求和排期意图澄清。它在启动模型和打开数据库前返回。排期参数通过 `start` 提交，确认通过 `confirm` 绑定当前提案和选项。

```powershell
python -m src.main chat '你好'
python -m src.main chat '帮我写一段短视频发布文案'
python -m src.main chat '帮我排个期'
python -m src.main chat '帮我给视频排个期'
```

用 `python -m src.main --db demo.db remember-preference DEMO-USER latest` 可显式记住「较晚的可用时段优先」，`earliest` 可改回较早优先。新请求使用同一个数据库和 `profile` 即可读取。偏好只排序已核验的备选时段。

真实模式安装依赖并配置可用模型。对于 OpenAI 接口，使用 `OPENAI_API_KEY` 或 `SCHEDULE_API_KEY`。自定义地址使用 `SCHEDULE_BASE_URL` 和该服务自己的 `SCHEDULE_API_KEY`。仅连接本机 loopback 时可省略密钥。端点若支持 JSON 对象模式，可设置 `SCHEDULE_JSON_MODE=object`，程序仍逐项校验返回。

```powershell
python -m pip install -r requirements.txt
# 按实际端点设置 SCHEDULE_MODEL 等进程环境变量
python -m src.benchmark --mode langchain --output examples/benchmark.model.json
```

本地验证使用 Qwen2.5-0.5B-Instruct 的 Q4_K_M GGUF 和 llama.cpp b12110。模型和运行时单独安装。把示例中的模型路径换成实际位置后启动服务。

```powershell
$modelPath = 'C:\models\qwen2.5-0.5b-instruct-q4_k_m.gguf'
llama-server -m $modelPath --host 127.0.0.1 --port 18765 --alias qwen2.5-0.5b-instruct-q4-k-m -c 4096 -np 1
```

```powershell
$env:SCHEDULE_BASE_URL = 'http://127.0.0.1:18765/v1'
$env:SCHEDULE_MODEL = 'qwen2.5-0.5b-instruct-q4-k-m'
$env:SCHEDULE_JSON_MODE = 'schema'
$env:NO_PROXY = '127.0.0.1,localhost'
python -m src.benchmark --mode langchain --baseline examples/benchmark.model-before.json
```

本地模型和服务端来自公开发行。 [Qwen 模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) · [llama.cpp b12110](https://github.com/ggml-org/llama.cpp/releases/tag/b12110)

离线评测覆盖 15 个案例。真实模型评测覆盖原时段可用、原时段占用和改期意愿待澄清三个案例，另跑一次关闭上下文的对照。报告含输入、工具返回、决策 Trace、数据库断言、Rubric 和模型用量。评测进程以退出码报告失败。

每次评估最多 8 次模型决策，每次请求最多 180 个输出 token。保存、状态恢复和缺少必填字段的检查由本地代码完成。更换兼容模型或端点后，用同一 benchmark 检查效果和用量。报告中的 `decision_usage` 汇总评估、澄清后的重跑和对照调用。计费可按端点的输入、输出单价乘以对应实测 token，裁判用量另计。

项目验收门槛为结构化 Rubric 平均至少 **80/100**，同时要求业务、重规划、ReAct 循环和上下文断言全部通过，且评估错误为零。报告里的 `verdict` 给出分数、门槛和各项通过条件。

| 示例 | 输入 | 实际输出 |
| --- | --- | --- |
| 正常排期，离线 | [原始请求](examples/normal.input.json) | [提案和确认后草稿](examples/normal.output.json) |
| 缺素材授权，离线 | [原始请求](examples/missing.input.json) | [追问字段和停止路线](examples/missing.output.json) |
| 保存工具失败，离线 | [请求与故障注入](examples/tool-failure.input.json) | [失败状态和零写入结果](examples/tool-failure.output.json) |
| 原时段占用，真实模型 | [请求和档期快照](examples/conflict.input.json) | [重规划、Top‑3、提案和确认后草稿](examples/conflict.output.json) |
| 改期意愿待澄清，真实模型 | [省略 flexible 的请求](examples/clarification.input.json) | [先追问，补答后生成提案](examples/clarification.output.json) |

正常和缺资料用例可用 `python -m src.main --db <新数据库文件> start <输入文件>` 重放。工具失败在离线 benchmark 中以 `save_failure` 重放。真实模型示例用 `python -m src.benchmark --mode langchain` 重放。基准在冲突场景模拟明确确认选项 1，在澄清场景模拟补充 `flexible=true` 后重新评估。

```powershell
python -m src.benchmark --mode offline --baseline examples/benchmark.offline.json
python -m src.benchmark --mode offline --human-scores examples/human-ratings.template.json
# 可选语义裁判，使用同一端点上可调用的模型
python -m src.benchmark --mode langchain --judge-model $env:SCHEDULE_MODEL
```

人评模板初始为空，填入实际评分后运行。报告会区分结构化机评、可选模型裁判和人工评分。历史评测结果及其适用范围见 [评估](docs/03-evaluation.md)。

[需求](docs/01-requirements.md) · [方案和字段](docs/02-solution.md) · [AI 使用](docs/04-ai-usage.md) · [复盘](docs/05-retrospective.md)
