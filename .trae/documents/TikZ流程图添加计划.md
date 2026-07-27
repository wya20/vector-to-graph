# 添加TikZ流程图计划

## 目标
在论文中添加系统架构流程图，使用TikZ绘制

## 当前状态
- Architecture Overview章节使用文字列表（第72-91行）
- 需要替换为可视化流程图

## 系统架构分析
5个核心组件：
1. Query Router（路由分类）
2. Vector Retrieval（向量检索）
3. Graph Retrieval（图谱检索）
4. Result Merger（结果合并）
5. API Server（API服务）

数据流：
- User Query → Router
- Router根据类型分发到Vector/Graph/Both
- 各检索结果 → Merger
- Merger → Results

## 执行步骤

### Step 1: 添加TikZ宏包
在导言区（\documentclass后）添加：
```latex
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc, fit}
```

### Step 2: 替换Architecture Overview内容
- 移除当前的enumerate列表
- 添加\begin{tikzpicture}环境
- 绘制节点：User、Router、Vector Search、Graph Search、Merger、Results
- 添加箭头表示数据流
- 添加路由分支逻辑（条件判断）

### Step 3: 节点布局设计
```
       ┌─────────────┐
       │  User Query │
       └──────┬──────┘
              │
              ▼
       ┌─────────────┐
       │   Router    │◄────── VECTOR_ONLY / GRAPH_ONLY / HYBRID
       └──────┬──────┘
              │
       ┌──────┴──────┐
       │             │
       ▼             ▼
┌─────────────┐ ┌─────────────┐
│   Vector    │ │   Graph    │
│   Search    │ │   Search   │
└──────┬──────┘ └──────┬──────┘
       │             │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │   Merger    │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │   Results   │
       └─────────────┘
```

### Step 4: 验证
- 确保LaTeX语法正确
- 图形不超出页面边界
- 文字标签清晰可读

## 修改文件
- papers/vector_to_graph_paper.tex

## 风险评估
- 低风险：TikZ是标准LaTeX包，arXiv支持
- 可能的样式问题可以通过调整参数解决
