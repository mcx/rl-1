import time
from os import path
import os

import submitit
import torch

from redq import main as main_redq, parser as parser_redq

# executor is the submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder="REDQ_log")

# set timeout in min, and partition for running the job
executor.update_parameters(timeout_min=1200, slurm_partition="train",
                           gpus_per_node=5, cpus_per_task=96)
jobs = []
exp_names = []
seed_list = [1, 42, 1988]
use_avg_pooling = [True, False]
shared_mapping = [True, False]

envs = []

from mj_envs.envs import *
import gym

envs = [k for k in gym.envs.registration.registry.env_specs.keys() if
        k.startswith("visual") and k != "visual_kitchen-v3"]

for env in envs:
    for _shared_mapping in shared_mapping:
        for _use_avg_pooling in use_avg_pooling:
            use_avg_pooling_str = ["avg_pooling"] if _use_avg_pooling else []
            shared_mapping_str = ["shared_mapping"] if _shared_mapping else []
            for seed in seed_list:
                exp_name = "_".join(
                    ["SUBMITIT", env, "seed",
                     str(seed)] + use_avg_pooling_str + shared_mapping_str)
                flags = ["--config",
                         f"redq_configs_pixels/generic.txt",
                         "--env_name", env,
                         "--seed", str(seed),
                         "--exp_name", exp_name,
                         "--collector_devices", "cuda:1", "cuda:2", "cuda:3",
                         "cuda:4", ]
                if _use_avg_pooling:
                    flags += ["--use_avg_pooling"]
                if _shared_mapping:
                    flags += ["--shared_mapping"]

                config = parser_redq.parse_args(flags)
                job = executor.submit(main_redq, config)
                print("flags:", flags, "\n\njob id: ", job.job_id, "\n\nexp_name: ", exp_name)  # ID of your job
                jobs.append(job)
                exp_names.append(exp_name)
                time.sleep(10)

for job, exp_name in zip(jobs, exp_names):
    output = job.result()  # waits for completion and returns output
    folder = output[0]
    torch.save(output[1:], path.join(folder, f"dump_{exp_name}.t"))