#import sys
#import os
#from main import main
import subprocess


def test_main():
    N = 10
    for deg in ['inp', 'deublur_gauss', 'delur_uni', 'cs4', 'cs8', 'cs16', 'sr2', 'sr4']:
        for s in [0.2, 0.1, 0.04]:
            print(f"*** RUNNING '{deg}' TEST WITH s={s} ***", flush=True)
            image_folder = f'imagenet/{deg}/s_{s}_n_{N}'
            args = ['python', 'main.py', '--doc', 'celeba', '--config', 'imagenet.yml', '--degradation', deg, '-i', image_folder, '-n', str(N), '-s', str(s)]
            subprocess.run(args) 


if __name__ == '__main__':
    test_main()
