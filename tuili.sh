#!/bin/bash

# 配置信息
SOURCE_DIR="/data/wubin/wanx-code"
REPO_URL="https://github.com/wangbei1/experiment.git"
TEMP_DIR="/tmp/wanx-code-upload"
EXCLUDES=(
    "outputs"
    "outputs_videojam"
    "outputs_videojam_flowonly"
    "outputs_videojam_token"
    "wan_models"
    "wanx_t2v_250103.pth"
    "vae.pth"
)

# 创建临时工作目录
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

echo "=== 克隆仓库 ==="
git clone "$REPO_URL" .
echo "仓库克隆完成"

# 设置仓库特定的用户信息（可选）
GIT_USER_NAME="Your Name"
GIT_USER_EMAIL="your.email@example.com"
git config user.name "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"

echo "=== 复制文件（排除指定目录）==="
rsync -av --progress "$SOURCE_DIR/" . \
    --exclude=".git" \
    --exclude=".gitignore" \
    $(printf -- "--exclude=%s " "${EXCLUDES[@]}")

echo "=== 检查排除结果 ==="
for exclude in "${EXCLUDES[@]}"; do
    if [ -e "$exclude" ]; then
        echo "警告: 排除目录 '$exclude' 仍然存在，请检查"
    else
        echo "✓ $exclude 已成功排除"
    fi
done

echo "=== 添加文件到Git ==="
git add --all .
git status

echo "=== 提交更改 ==="
read -p "请输入提交信息: " commit_message
if [ -z "$commit_message" ]; then
    commit_message="更新 wanx-code 代码"
fi

git commit -m "$commit_message"

echo "=== 推送到远程仓库 ==="
git push origin main

echo "=== 清理临时目录 ==="
rm -rf "$TEMP_DIR"

echo "=== 上传完成 ==="
echo "仓库地址: $REPO_URL"