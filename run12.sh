#!/bin/bash

# 这个脚本会删除当前目录下所有符合 run<数字>.py 格式的文件
# 例如: run1.py, run2.py, run10.py

echo "正在查找并准备删除 run<数字>.py 格式的文件..."

# 为了防止在没有匹配项时出错，我们先检查文件是否存在
# shopt -s nullglob 使得在没有匹配时，模式会扩展为空列表
shopt -s nullglob
files=(run[0-9]*.py)

if [ ${#files[@]} -gt 0 ]; then
    echo "将要删除以下文件:"
    # 列出将要被删除的文件
    printf "%s\n" "${files[@]}"
    
    # 实际执行删除操作，-v 参数会显示被删除的文件名
    rm -v "${files[@]}"
    
    echo "删除完成。"
else
    echo "没有找到符合 'run<数字>.py' 格式的文件。"
fi