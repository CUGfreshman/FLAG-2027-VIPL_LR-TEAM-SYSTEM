"""Train the fixed competition-English projection from prepared features."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Prepared rows.csv/face.npy/voice.npy directory')
    parser.add_argument('--output', required=True, help='Run directory; checkpoints are written directly here and recovery uses its latest checkpoint')
    args = parser.parse_args()
    from flag2027.triplet.training import train
    train(args.data, args.output)


if __name__ == '__main__':
    main()
