import os
import re
from datetime import datetime
from huggingface_hub import HfApi, create_repo

# ==== 配置区 ====
# 相对路径（相对你运行此脚本时的工作目录）
CKPT_DIR = "outputs/latest-run/checkpoints"

# 目标仓库
REPO_ID = "wangbei1/qianyi"   # e.g. "username/repo_name"
REPO_TYPE = "model"           # 通常模型仓库用 "model"

# 读取 token（优先环境变量 HF_TOKEN，找不到再用变量 TOKEN_FALLBACK）
TOKEN_FALLBACK = None  # 你也可以临时在这里放一个 token 字符串，但不推荐
token = "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"

if not token:
    raise RuntimeError(
        "没有找到 Hugging Face 访问令牌。请先设置环境变量 HF_TOKEN='your_token_here'。"
    )

# ==== 1) 找出最大 step 的 ckpt ====
pattern = re.compile(r"^dfot-step[=-](\d+)\.ckpt$")
max_step = -1
best_file = None

if not os.path.isdir(CKPT_DIR):
    raise FileNotFoundError(f"找不到目录：{CKPT_DIR}")

for fname in os.listdir(CKPT_DIR):
    m = pattern.match(fname)
    if m:
        step = int(m.group(1))
        if step > max_step:
            max_step = step
            best_file = fname

if best_file is None:
    raise FileNotFoundError(
        f"在 {CKPT_DIR} 下没有找到形如 dfot-step=*.ckpt 或 dfot-step-*.ckpt 的文件。"
    )

local_path = os.path.join(CKPT_DIR, best_file)
print(f"将要上传的本地文件：{local_path}（step={max_step}）")

# ==== 2) 生成上传文件名（把 = 改成 -，并追加时间戳）====
# 原始名例如：dfot-step=50000.ckpt  -> dfot-step-50000_1023_2026.ckpt
ts = datetime.now().strftime("%m%d_%H%M")  # 例如 "1023_2026"
repo_basename = best_file.replace("=", "-")
repo_name_with_time = repo_basename.replace(".ckpt", f"_{ts}.ckpt")

# 如果你想把 ckpt 放到子目录，比如 "checkpoints/" 下：
# repo_path = f"checkpoints/{repo_name_with_time}"
# 否则直接放根目录：
repo_path = repo_name_with_time

print(f"上传到仓库的路径：{REPO_ID}/{repo_path}")

# ==== 3) 创建（或复用）仓库并上传 ====
api = HfApi(token=token)

# 如果仓库已存在会复用；不存在则创建
create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=True, exist_ok=True, token=token)

api.upload_file(
    path_or_fileobj=local_path,
    path_in_repo=repo_path,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)

print("✅ 文件上传成功！")
