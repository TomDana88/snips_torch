import sys
import os
from main import main


def test_main():
    for deg in ['deblur_uni', 'sr2', 'sr4']:
        n = 8
        image_folder = f'celeba_deg_{deg}_n_{n}'
        sys.argv = ['./main.py', '--doc', 'celeba', '--config', 'celeba.yml',
                    '--degradation', deg, '-i', image_folder, '-n', str(n)]
        main()


if __name__ == '__main__':
    test_main()
