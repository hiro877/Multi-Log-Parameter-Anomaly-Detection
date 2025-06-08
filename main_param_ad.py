import argparse
from models.anomaly_detection.parameter.pipeline import ParameterADPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    pipeline = ParameterADPipeline(args.config)
    pipeline.train()
    pipeline.evaluate()


if __name__ == "__main__":
    main()
