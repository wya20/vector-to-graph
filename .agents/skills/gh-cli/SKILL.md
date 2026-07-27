---
name: gh-cli
description: GitHub CLI (gh) 常用操作速查。当需要操作 GitHub 仓库、管理 Issue/PR、查看 CI 状态、创建 Release 时使用。
---

# gh CLI 常用操作速查

## 认证
- `gh auth status` — 查看当前登录状态

## 仓库操作
- `gh repo view <owner/repo>` — 查看仓库详情
- `gh repo clone <owner/repo> [<dir>]` — 克隆仓库
- `gh repo fork [<owner/repo>]` — Fork 仓库
- `gh repo create <name> [--public|--private] [--clone]` — 创建仓库

## Pull Request
- `gh pr list [-R <owner/repo>] [-s <state>] [-l <labels>]` — 列出 PR
- `gh pr view <number> [-R <owner/repo>] [--comments]` — 查看 PR 详情
- `gh pr view <number> --json <fields>` — 以 JSON 格式查看（可解析）
- `gh pr create [--title <t>] [--body <b>] [--base <b>]` — 创建 PR
- `gh pr checkout <number>` — 检出 PR 到本地
- `gh pr merge <number> [--merge|--squash|--rebase]` — 合并 PR
- `gh pr review <number> --approve|--comment|--request-changes` — 审查 PR

## Issue
- `gh issue list [-R <owner/repo>] [-s <state>] [-l <labels>]` — 列出 Issue
- `gh issue view <number> [--comments]` — 查看 Issue
- `gh issue create [--title <t>] [--body <b>] -l <labels>` — 创建 Issue
- `gh issue close <number>` — 关闭 Issue
- `gh issue comment <number> --body <text>` — 评论 Issue

## Actions / CI
- `gh run list [-w <workflow-name>] [-L <limit>]` — 列出工作流运行
- `gh run view <run-id> [--log]` — 查看运行详情/日志
- `gh run watch <run-id>` — 实时查看运行日志
- `gh workflow list` — 列出所有工作流
- `gh workflow run <workflow> [--ref <branch>]` — 手动触发工作流

## Release
- `gh release list [-L <limit>]` — 列出 Releases
- `gh release view <tag>` — 查看 Release 详情
- `gh release create <tag> [--title <t>] [--notes <n>] <files>` — 创建 Release

## 通用参数
- `-R <owner/repo>` — 指定目标仓库（不指定时默认为当前仓库）
- `--json <fields>` — 输出 JSON 格式，配合 `-q <jq表达式>` 做字段过滤
- `-q <expression>` — 用 jq 语法过滤 JSON 输出
- `-t <title>`, `-b <body>` — 标题和内容
- `-H <branch>` — 源分支（PR 用）
- `-B <branch>` — 目标分支（PR 用）

## 实用技巧
- 管道组合：`gh pr list --json number,title -q '.[] | "\(.number): \(.title)"'`  # 格式化输出
- 结合 `|` 和 `xargs` 做批量操作，如：`gh pr list --json number -q '.[].number' | ForEach-Object { gh pr view $_ }`
- 查看完整字段：`gh pr view <number> --json <TAB>`（按 Tab 可提示可用字段）