kvpress_path=$(python -c "import os; import kvpress; kvpress_dir = os.path.dirname(kvpress.__file__); print(kvpress_dir)")
echo $kvpress_path
# kvpress_path=/root/miniconda3/lib/python3.10/site-packages/kvpress
kvpress_path=/opt/conda/lib/python3.10/site-packages/kvpress
cp /modelopsnas/modelops/468440/kvpress/kvpress/presses/*.py $kvpress_path/presses
cp /modelopsnas/modelops/468440/kvpress/kvpress/presses/adathink_press.py $kvpress_path/presses
cp /modelopsnas/modelops/468440/kvpress/kvpress/__init__.py $kvpress_path
cp /modelopsnas/modelops/468440/kvpress/kvpress/pipeline.py $kvpress_path