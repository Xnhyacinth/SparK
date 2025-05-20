# -*- coding: utf-8 -*-
from pypai.job import PythonJobBuilder, PytorchJobBuilder
from pypai.conf import ExecConf, KMConf, GpuType, NasStoreConf
from aistudio_common.openapi.models import DataStore
from pypai.conf.retry_strategy import RetryStrategy,RetryPolicy



def main():
    image = "reg.docker.alibaba-inc.com/aii/aistudio:aistudio-173030369-1220582068-1740889140816"

    #rs = RetryStrategy(retry_policy=RetryPolicy.ON_FAILURE, max_attempt=3)
    #km_conf = KMConf(
    #    image=image,
    #    retry_strategy=rs,
    #)
    km_conf = KMConf(image=image)

    node_num = 1
    # master = ExecConf(cpu=96, num=1, shared_memory=1048576, memory=1048576, gpu_num=8, gpu_percent=100, disk_m=1048576)
    # master = ExecConf(cpu=48, num=1, shared_memory=1048576, memory=1048576, gpu_num=4, gpu_percent=100, disk_m=1048576)
    master = ExecConf(cpu=24, num=1, shared_memory=65536, memory=256000, gpu_num=2, gpu_percent=100, disk_m=1048576)
    # master = ExecConf(cpu=24, num=1, shared_memory=65536, memory=256000, gpu_num=1, gpu_percent=100, disk_m=1048576)
    if node_num - 1 > 0:
        worker = ExecConf(cpu=48, num=node_num-1, shared_memory=65536, memory=512000, gpu_num=8, gpu_percent=100, disk_m=204800)
    else:
        worker = None
  
    nas_input = DataStore(mount_point="/modelopsnas/modelops/468440", store_name="ais-nas-tech.risk", sub_path="/")

    job = PytorchJobBuilder(source_root='./', 
                           command='bash run5.sh',
                           main_file='',
                           master=master,
                           worker=worker,
                           km_conf=km_conf,
                           k8s_app_name="riskmining",
                           tag="type=5-kv_3_t1.0,basemodel=llama3-8b",
                           k8s_priority="low",
                           platform="kubemaker",
                           rdma=1, #必填，且值必须等于1
                           hostnetwork=True, #必填，填true
                           data_stores=[nas_input]
                           )
    
    
    labels = dict()
    labels['limited_time'] = 60 * 24 * 30  # 分钟
    job.labels = labels
    job.run(dima='2025010300106831832')
    # job.run(dima='2024082100104214924')


if __name__ == '__main__':
    main()