import os
import re
from datetime import datetime
from huggingface_hub import HfApi, create_repo

# ===== 配置 =====
CKPT_DIR = "outputs/latest-run/checkpoints"
LOSS_PATH = "outputs/latest-run/loss.txt"
REPO_ID = "wangbei1/qianyi"
REPO_TYPE = "model"

# 令牌：优先从环境变量取，避免硬编码
token = "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"
if not token:
    raise RuntimeError("请先 export HF_TOKEN=你的token 再运行。")

# ===== 1) 选出 step 最大的 ckpt =====
pattern = re.compile(r"^dfot-step[=-](\d+)\.ckpt$")
if not os.path.isdir(CKPT_DIR):
    raise FileNotFoundError(f"找不到目录：{CKPT_DIR}")

max_step = -1
best_file = None
for fname in os.listdir(CKPT_DIR):
    m = pattern.match(fname)
    if m:
        step = int(m.group(1))
        if step > max_step:
            max_step = step
            best_file = fname

if best_file is None:
    raise FileNotFoundError(f"在 {CKPT_DIR} 下没有找到 dfot-step=*.ckpt / dfot-step-*.ckpt")

ckpt_local = os.path.join(CKPT_DIR, best_file)
print(f"将要上传的 ckpt：{ckpt_local} (step={max_step})")

# ===== 2) 生成统一时间戳，并构造上传名 =====
ts = datetime.now().strftime("%m%d_%H%M")  # 例如 1023_2026

# ckpt：把 '=' 改成 '-'，并追加时间戳
ckpt_repo_name = best_file.replace("=", "-").replace(".ckpt", f"_{ts}.ckpt")
ckpt_repo_path = ckpt_repo_name                     # 放根目录；如需子目录：f"checkpoints/{ckpt_repo_name}"

# loss.txt：也带时间戳
if not os.path.isfile(LOSS_PATH):
    raise FileNotFoundError(f"找不到 loss 日志文件：{LOSS_PATH}")
loss_repo_name = f"loss_{ts}.txt"
loss_local = LOSS_PATH
loss_repo_path = loss_repo_name                     # 放根目录；如需子目录：f"logs/{loss_repo_name}"

print(f"上传到仓库的 ckpt 路径：{REPO_ID}/{ckpt_repo_path}")
print(f"上传到仓库的 loss 路径：{REPO_ID}/{loss_repo_path}")

# ===== 3) 创建/复用仓库并上传 =====
api = HfApi(token=token)
create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=True, exist_ok=True, token=token)

# # 上传 ckpt
# api.upload_file(
#     path_or_fileobj=ckpt_local,
#     path_in_repo=ckpt_repo_path,
#     repo_id=REPO_ID,
#     repo_type=REPO_TYPE,
# )
# print("✅ ckpt 上传成功")

# 上传 loss.txt
api.upload_file(
    path_or_fileobj=loss_local,
    path_in_repo=loss_repo_path,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)
print("✅ loss.txt 上传成功")
