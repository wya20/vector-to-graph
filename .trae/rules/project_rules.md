# 工具选择优先级

## GitHub 任务

遇到与 GitHub 相关的任务时，遵循以下优先顺序：
1. **优先使用 `gh` CLI** 命令行工具（已安装 v2.92.0）
2. 仅当 `gh` CLI 无法完成某个操作时，才考虑使用 GitHub MCP 工具

> 详细命令速查 → `.agents/skills/gh-cli/SKILL.md`（按需加载）

## 联网搜索（Tavily MCP）

| 场景 | 工具 |
|------|------|
| 搜互联网 / 技术文档 / 错误方案 | `tavily_search` |
| 爬取网页内容 / 取 GitHub README 或文档页 | `tavily_extract` / `tavily_crawl` |
| 深度调研 | `tavily_research` |

## TRAE 内置工具的使用边界

| 仍用内置工具 | 说明 |
|------|------|
| 代码语义搜索 | `SearchCodebase` |
| 修改文件 | `Write` / `SearchReplace` |
| 读小文件 | `Read` |
| 搜代码/文本 | `Grep` |
| 按文件名查找 | `Glob` |
| 列目录 | `LS` |

| 避免内置工具的场景 | 原因 / 替代 |
|------|------|
| 读大文件 | `Read` 全文进上下文 → 改用 `RunCommand` 执行 `Get-Content <file> -Head N` 分步读取 |
| 搜代码/文本（大量结果） | 结果过多撑爆上下文 → 改用 `RunCommand` 执行 `rg "<pattern>"` |

## 项目专用 MCP 工具

| 工具 | 场景 |
|------|------|
| `vtg_query` | 按语义搜索代码，替代直接读文件 |
| `vtg_get_graph` | 探索代码关系（函数调用、类继承、导入） |
| `vtg_index` | 修改代码后更新知识图谱 |
| `vtg_get_status` | 检查服务状态 |