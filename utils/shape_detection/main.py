import shape
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='Shape_Detection_By_DTW')
    parser.add_argument('detection_for', type=str, help='which shape detection to run')
    result = parser.parse_args()
    return result


def main():
    args = get_arguments()
    shape.shape_detection_by_dtw_algorithm(args.detection_for)


if __name__ == '__main__':
    main()
