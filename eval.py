import argparse
from mainloop import mainloop

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')

    # parse configs
    args = parser.parse_args()
    mainloop('val', args)


