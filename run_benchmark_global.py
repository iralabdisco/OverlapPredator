import os, shutil
from multiprocessing import Pool
from pathlib import Path
PY3="python3"

GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

BENCHMARK_DIR = "/benchmark/point_clouds_registration_benchmark/dataset_voxelgrid_0.025"
MODEL = "3DMatch"
N_POINTS = 5000
RESULTS_DIR = f"/benchmark/experiments/OverlapPredator/{MODEL}/features"

base_command = ( f'{PY3}' + f' scripts/predator_benchmark.py {MODEL} {N_POINTS}')

problem_txts = ['kaist/urban05_global.txt',
                'eth/apartment_global.txt',
                'eth/gazebo_summer_global.txt',
                'eth/gazebo_winter_global.txt',
                'eth/hauptgebaude_global.txt',
                'eth/plain_global.txt',
                'eth/stairs_global.txt',
                'eth/wood_autumn_global.txt',
                'eth/wood_summer_global.txt',
                'tum/long_office_household_global.txt',
                'tum/pioneer_slam_global.txt',
                'tum/pioneer_slam3_global.txt',
                'planetary/box_met_global.txt',
                'planetary/p2at_met_global.txt',
                'planetary/planetary_map_global.txt']

pcd_dirs = ['kaist/urban05/',
            'eth/apartment/',
            'eth/gazebo_summer/',
            'eth/gazebo_winter/',
            'eth/hauptgebaude/',
            'eth/plain/',
            'eth/stairs/',
            'eth/wood_autumn/',
            'eth/wood_summer/',
            'tum/long_office_household/',
            'tum/pioneer_slam/',
            'tum/pioneer_slam3/',
            'planetary/box_met/',
            'planetary/p2at_met/',
            'planetary/p2at_met/']

feature_dirs = ['kaist/urban05/',
            'eth/apartment/',
            'eth/gazebo_summer/',
            'eth/gazebo_winter/',
            'eth/hauptgebaude/',
            'eth/plain/',
            'eth/stairs/',
            'eth/wood_autumn/',
            'eth/wood_summer/',
            'tum/long_office_household/',
            'tum/pioneer_slam/',
            'tum/pioneer_slam3/',
            'planetary/box_met/',
            'planetary/p2at_met/',
            'planetary/planetary_map/']

commands = []

for problem_txt, pcd_dir, feature_dir in zip(problem_txts, pcd_dirs, feature_dirs):
    full_command = (base_command +
                    f' --input_txt={BENCHMARK_DIR}/{problem_txt}' +
                    f' --input_pcd_dir={BENCHMARK_DIR}/{pcd_dir}' +
                    f' --output_dir={RESULTS_DIR}/{feature_dir}')

    problem_name = Path(problem_txt).stem
    time_command = f'command time --verbose -o {RESULTS_DIR}/{problem_name}_time.txt ' + full_command
    nvidia_command = (f'nvidia-smi --query-gpu=timestamp,memory.used -i 0 --format=csv -lms 1 > {RESULTS_DIR}/{problem_name}_memory.txt')

    full_command_stats = f'parallel -j2 --halt now,success=1 ::: \'{time_command}\' \'{nvidia_command}\''
    commands.append(full_command_stats)

answer = input(f"Delete previous {RESULTS_DIR}? [Y/N] ")
if answer != "Y":
    print("Quitting...")
    exit()

# delete and recreate result directory
shutil.rmtree(RESULTS_DIR, ignore_errors=True)
os.makedirs(RESULTS_DIR)

# save config in result directory
txt_commands = os.path.join(RESULTS_DIR, "readme.md")
with open(txt_commands, 'w') as f:
    for item in commands:
        f.write("%s\n" % item)

pool = Pool(1)
pool.map(os.system, commands)