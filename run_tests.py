import subprocess

def test_main():
    N = 8
    NOISE_TYPE = 'speckle'
    for deg in ['cs8']:
        print(f"*** RUNNING '{deg}' with noise={NOISE_TYPE} ***", flush=True)
        image_folder = f'celeba/speckle/{deg}'
        args = f'python main.py --config celeba.yml --doc celeba -d {deg} -i {image_folder} -n 8 --noise_type {NOISE_TYPE} --overwrite --verbose error'
        subprocess.run(args.split())


if __name__ == '__main__':
    test_main()
