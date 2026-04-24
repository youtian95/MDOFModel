# 开发文档

## 开发安装

克隆仓库后，在项目根目录执行以下命令将包以可编辑模式安装到当前环境：

```bash
pip install -e .
```

---

## 打包流程

确保已安装打包工具：

```bash
pip install build twine
```

1. **更新版本号**

   修改 `pyproject.toml` 中的 `version` 字段：

   ```toml
   [project]
   version = "x.y.z"
   ```

   同时在 `CHANGELOG.md` 中记录本次变更内容。

2. **构建包**

   在项目根目录（`pyproject.toml` 所在目录）执行：

   ```bash
   python -m build
   ```

   构建完成后会在 `dist/` 目录下生成两个文件：
   - `MDOFModel-x.y.z.tar.gz`（源码包）
   - `MDOFModel-x.y.z-py3-none-any.whl`（wheel 包）

3. **上传到 PyPI**

   ```bash
   python -m twine upload dist/*
   ```
