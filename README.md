# Vector-to-Graph

**结合向量检索的速度和准确性 + Graphify 的关系理解和可解释性，打造更适合代码和技术文档场景的混合检索引擎。**

## 特性

- 🚀 **混合检索**：结合向量检索和图谱检索的优点
- 📊 **自动建图**：基于 tree-sitter 自动提取代码结构关系
- 🔍 **智能路由**：自动判断查询类型，选择最优检索策略
- 📈 **可解释性**：每个结果都标注来源和置信度
- 🐳 **容器化部署**：一行命令启动完整服务

## 架构

```
用户问题 → 智能路由 → 并行查询
                         ↓
              ┌─────────┴─────────┐
              ↓                   ↓
        向量检索            图谱检索
        (快速初筛)         (关系理解)
              ↓                   ↓
              └─────────┬─────────┘
                        ↓
                  结果合并
                   (加权排序)
                        ↓
                    生成答案
```

## 快速开始

### Docker Compose（推荐）

```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/vector-to-graph.git
cd vector-to-graph

# 启动服务
docker-compose up -d

# API 文档：http://localhost:8000/docs
```

### 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Qdrant（需要 Docker）
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 启动服务
uvicorn src.api.server:app --reload --port 8000
```

## API 使用

### 索引文档

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"path": "./data/raw"}'
```

### 查询

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "装饰器函数是怎么实现的？"}'
```

### 获取图谱

```bash
curl http://localhost:8000/graph
```

## 支持的文件类型

| 类型 | 扩展名 |
|------|--------|
| Python | .py |
| JavaScript | .js, .ts, .jsx, .tsx |
| Markdown | .md |

## AI 助手集成

Vector-to-Graph 支持通过 MCP (Model Context Protocol) 与各种 AI 编码助手集成。

### 支持的平台

| AI 助手 | 配置位置 | 安装方式 |
|---------|----------|----------|
| Claude Code / Desktop | [integrations/claude-desktop/](integrations/claude-desktop/) | `claude mcp add --transport http vtg http://localhost:8000/mcp` |
| Trae | [integrations/trae/](integrations/trae/) | 复制到 `.trae/mcp.json` |
| Cursor | [integrations/cursor/](integrations/cursor/) | 复制到 `.cursor/mcp.json` |
| OpenCode | [integrations/opencode/](integrations/opencode/) | 复制到 `.opencode/mcp.json` |
| OpenClaw | [integrations/openclaw/](integrations/openclaw/) | 复制到 `.claw/mcp.json` |
| Windsurf | [integrations/windsurf/](integrations/windsurf/) | 复制到 `.windsurf/mcp.json` |

详细说明请查看 [integrations/README.md](integrations/README.md)。

### 快速集成

1. 启动 vector-to-graph 服务：
   ```bash
   docker-compose up -d
   ```

2. 启动 MCP Server：
   ```bash
   uv run python -m src.mcp.server
   ```

3. 在 AI 助手中添加 MCP Server（以 Claude Code 为例）：
   ```bash
   claude mcp add --transport http vtg http://localhost:8000/mcp
   ```

### 可用工具

- `vtg_index(path)`: 索引代码/文档
- `vtg_query(question, limit)`: 查询混合搜索引擎
- `vtg_get_graph()`: 获取知识图谱结构
- `vtg_get_status()`: 查看服务状态

## 技术栈

- **向量检索**：Qdrant + Sentence-Transformers
- **图谱构建**：NetworkX + Tree-sitter
- **API 服务**：FastAPI + Uvicorn
- **社区检测**：Louvain/Leiden 算法
- **AI 集成**：MCP (Model Context Protocol)

## License

MIT License - 详见 [LICENSE](LICENSE) 文件