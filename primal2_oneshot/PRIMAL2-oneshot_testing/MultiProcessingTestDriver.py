import warnings
import os
import subprocess
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=Warning)


def get_map_names(env_path, result_path, planner, resume_testing):
    print('loading testing env...')

    if not env_path.endswith('/'):
        env_path += '/'
    if not result_path.endswith('/'):
        result_path += '/'

    valid = []
    for name in os.listdir(env_path):
        if not name.endswith('.npy'):
            continue

        if resume_testing:
            env_name = name[:-4]
            result_file = f"{result_path}{env_name}_oneshot{planner}.txt"
            if os.path.exists(result_file):
                continue

        valid.append(name)

    print(f'There are {len(valid)} remaining tests detected')
    return valid


def run_1_test(args, name):
    script_path = os.path.join(os.path.dirname(__file__), "TestingEnv.py")

    cmd = [
        "python",
        script_path,
        "-p", args.planner,
        "-n", name,
        "--env_path", args.env_path,
        "--result_path", args.result_path
    ]

    try:
        subprocess.run(cmd, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {name}")


def run_tests(args):
    maps = get_map_names(args.env_path, args.result_path, args.planner, args.resume_testing)

    workers = min(args.num_worker, cpu_count())
    print(f"start testing with {workers} workers...")

    with Pool(workers) as pool:
        func = partial(run_1_test, args)
        list(pool.imap_unordered(func, maps))  # ✅ efficient + no busy loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", default="./oneshot_testing_result/")
    parser.add_argument("--env_path", default="./test_envs/")
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument("-r", "--resume_testing", default=True)
    parser.add_argument("-p", "--planner", default="mstar")

    args = parser.parse_args()

    run_tests(args)