kvpress_path=$(python -c "import os; import kvpress; kvpress_dir = os.path.dirname(kvpress.__file__); print(kvpress_dir)")
echo $kvpress_path
# kvpress_path=/root/miniconda3/lib/python3.10/site-packages/kvpress
# kvpress_path=/home/tiger/.local/lib/python3.11/site-packages/kvpress
cp kvpress0/presses/*.py $kvpress_path/presses
cp kvpress0/presses/adathink_press.py $kvpress_path/presses
cp kvpress0/__init__.py $kvpress_path
cp kvpress0/pipeline.py $kvpress_path