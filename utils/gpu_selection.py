#!/usr/bin/env python3

'''
@Time   : 2018-02-28 21:19:36
@Author : su.zhu
@Desc   : 
'''

import os
import sys
if '/cm/local/apps/cuda/libs/current/pynvml' in sys.path:
    sys.path.remove('/cm/local/apps/cuda/libs/current/pynvml')
import gpustat

def get_gpu_compute_rest(gpu_stats_list, gpu_id_list):
    device_compute_rest = {}
    for gpu_stat in gpu_stats_list:
        if gpu_stat['index'] not in gpu_id_list:
            continue
        memory_used = gpu_stat['memory.used']
        memory_total = gpu_stat['memory.total']
        memory_rest = 1 - float(memory_used)/memory_total
        utilization_gpu = gpu_stat['utilization.gpu']
        utilization_gpu_rest = 1 - float(utilization_gpu)/100
        device_compute_rest[gpu_stat['index']] = (utilization_gpu_rest, memory_rest)
    print("GPU device state {idx:(compute_rest, memory_rest)} : " , device_compute_rest)
    return device_compute_rest

def auto_select_gpu(assigned_gpu_id=None):
    gpu_id_list = os.getenv('CUDA_VISIBLE_DEVICES')
    gpu_stats_list = gpustat.new_query()
    if gpu_id_list == None:
        gpu_id_list = [g['index'] for g in gpu_stats_list]
    else:
        gpu_id_list = [int(value) for value in gpu_id_list.split(',')]

    if assigned_gpu_id != None:
        best = assigned_gpu_id
        if best >= len(gpu_id_list):
            print("WARNING: Manually selected gpu index is out of range!")
            best = 0
        gpu_name = gpu_stats_list[gpu_id_list[best]]['name']
    else:
        device_compute_rest = get_gpu_compute_rest(gpu_stats_list, gpu_id_list)
        device_first_level, device_second_level, device_third_level = {}, {}, {}
        for i in device_compute_rest:
            computeRestRate, memRestRate = device_compute_rest[i]
            if memRestRate > 0.3 and computeRestRate > 0.5:
                device_first_level[i] = computeRestRate
            elif memRestRate > 0.1 and memRestRate <= 0.3 and computeRestRate > 0.5:
                device_second_level[i] = memRestRate
            else:
                device_third_level[i] = memRestRate + computeRestRate

        if len(device_first_level) > 0:
            best = max(device_first_level.items(), key=lambda x: x[1])[0]
            print("INFO: Using the first level GPU card")
        else:
            if len(device_second_level) > 0:
                best = max(device_second_level.items(), key=lambda x: x[1])[0]
                print("WARNING: Using the second level GPU card")
            else:
                print("WARNING: Using the third level GPU card")
                best = max(device_third_level.items(), key=lambda x: x[1])[0]
        gpu_name = gpu_stats_list[best]['name']
        best = gpu_id_list.index(best)
    valid_gpus = [str(gpu_idx) for gpu_idx in gpu_id_list]
    return best, gpu_name, ','.join(valid_gpus)

