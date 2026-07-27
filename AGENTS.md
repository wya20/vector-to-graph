# Vector-to-Graph — 代码知识图谱 + 向量混合搜索

- 运行环境：Windows / Linux / macOS
- 语言：Python 3.10+
- 外部工具：gh CLI v2.92.0、Docker、Qdrant（向量数据库）、Tree-sitter（代码解析）
- 协议：MCP（Model Context Protocol）
- 包管理：pip / uv（pyproject.toml）
- 行为指南请参见同目录下的 `CLAUDE.md`

## 项目简介

将代码仓库转化为可查询的知识图谱，结合向量相似度搜索和图关系遍历，实现代码的语义理解。

核心流程：
1. **代码解析** — Tree-sitter 解析 AST，提取函数、类、导入关系
2. **向量索引** — 代码分块 → Embedding → Qdrant 存储
3. **图构建** — NetworkX 构建调用图、继承图、依赖图
4. **混合查询** — 向量相似度 + 图遍历融合排序

## 目录结构

```
vector-to-graph/
├── .trae/
│   ├── rules/
│   │   └── project_rules.md           # 工具选择优先级（每次对话自动注入）
│   └── documents/                     # 项目文档
├── .github/
│   └── workflows/                     # CI/CD（ci.yml / release.yml）
├── src/
│   ├── api/                           # REST API 服务
│   ├── mcp/                           # MCP 协议服务端
│   ├── graph_builder/                 # 图构建（解析器、社区检测、导出）
│   ├── vector_indexer/                # 向量索引（分块、嵌入、Qdrant）
│   ├── query_router/                  # 查询路由
│   ├── result_merger/                 # 结果融合
│   ├── sync_manager/                  # 文件同步（监听、事务日志）
│   └── utils/                         # 配置等工具
├── tests/                             # 测试套件
├── integrations/                      # 各 IDE 的 MCP 配置模板
├── experiments/                       # 实验脚本
├── papers/                            # 论文（LaTeX）
├── AGENTS.md                          # 本文件（项目级 Agent 行为规范）
├── CLAUDE.md                          # LLM 通用行为指南
├── pyproject.toml                     # 项目元数据与依赖
├── Dockerfile
└── docker-compose.yaml
```

## 编码规范

**Python 代码**
- 遵循 PEP 8，使用 4 空格缩进
- 类型注解：函数签名使用 type hints
- 模块 `__init__.py` 中显式导出公共 API（`__all__`）
- 文档字符串使用 Google 风格（Args / Returns / Raises）
- 配置通过 `src/utils/config.py` 统一管理，不硬编码

**错误处理**
- 网络请求和外部服务调用使用 `try/except` 包裹
- 自定义异常类统一放在模块内，继承自项目基类
- 日志使用 `logging` 模块，分 INFO / WARNING / ERROR 级别

**测试**
- 测试文件命名：`test_<module_name>.py`
- 使用 pytest 框架
- 外部依赖（Qdrant、Tree-sitter）的测试使用 mock 或标记为 integration

## 测试与验证

```powershell
# 运行全部测试
python -m pytest tests/ -v

# 运行单个模块测试
python -m pytest tests/test_chunker.py -v

# 启动 API 服务
python -m src.api.server

# 启动 MCP 服务
python -m src.mcp.server
```

## MCP 工具（Vector-to-Graph 专用）

| 工具 | 用途 |
|------|------|
| `vtg_query(question, limit)` | 混合搜索（向量 + 图） |
| `vtg_get_graph()` | 获取知识图谱结构 |
| `vtg_index(path)` | 索引文件或目录 |
| `vtg_get_status()` | 检查服务状态 |

使用规则：
- 回答代码架构问题前，优先用 `vtg_query` 搜索
- 探索代码关系（调用链、继承、导入）时用 `vtg_get_graph`
- 修改代码文件后，运行 `vtg_index` 更新图谱

## Never 规则

- Never 修改 AGENTS.md、CLAUDE.md，除非用户明确要求
- Never 硬编码 API Key、Token 或任何敏感凭据
- Never 提交 `.env` 或包含凭据的文件到 Git
- Never 删除 `src/` 下的核心模块文件
- Never 使用 `cmd.exe` 运行命令（使用 PowerShell）
- Never 在路径中使用反斜杠转义时省略引号包裹

## Git 提交规范

- 格式：`type(scope): description`
- 类型：`feat` / `fix` / `docs` / `style` / `refactor` / `test` / `chore`

## 多 Agent 协作

- 禁止 `git stash` — 不同的 Agent 可能 stash 掉彼此的改动
- 禁止切换分支 — 当前分支的工作由各 Agent 自行 commit
- 只操作分配给自己的文件，不修改其他 Agent 正在处理的部分