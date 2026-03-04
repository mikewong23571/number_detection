## Number Detection Demo (PyTorch + UV)

MNIST 手写数字识别 demo，使用 Python + PyTorch + `uv` 管理依赖。

### 本地运行

```bash
uv sync
uv run number-detection --epochs 1 --max-train-samples 10000 --max-test-samples 2000
```

默认会：

- 自动下载 MNIST 到 `data/`
- 在 `artifacts/mnist_cnn.pt` 保存模型
- 输出测试集准确率和示例预测结果

### 常用参数

```bash
uv run number-detection --help
```

示例（若本机有 CUDA）：

```bash
uv run number-detection --epochs 2 --device cuda
```

### Colab 运行

仓库支持直接 clone 到 Colab 执行，可参考 `notebooks/mnist_colab_demo.ipynb`。

同时提供配置文件 `colab-run.yaml`，可用于 config-driven 的 Colab bootstrap 流程。

核心命令：

```bash
git clone <YOUR_REPO_URL>
cd number_detection
pip install -q uv
uv sync
uv run number-detection --epochs 1 --device cuda
```

### 使用 colab-cli 指定 T4 并清理实例

```bash
# 申请 T4
npx --yes --package=git+https://github.com/mikewong23571/colab-vscode.git#main colab-cli -- assign add --variant GPU --accelerator T4

# 在分配到的 endpoint 上执行（将 <ENDPOINT> 替换成上一步输出）
npx --yes --package=git+https://github.com/mikewong23571/colab-vscode.git#main colab-cli -- exec <ENDPOINT> -- bash -lc "git clone <YOUR_REPO_URL> && cd number_detection && pip install -q uv && uv sync && uv run number-detection --epochs 1 --device cuda"

# 删除实例
npx --yes --package=git+https://github.com/mikewong23571/colab-vscode.git#main colab-cli -- assign rm <ENDPOINT>
```
