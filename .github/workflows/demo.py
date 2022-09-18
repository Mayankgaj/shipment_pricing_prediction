import logging
import os
from shipment.pipeline.pipeline import Pipeline


def main():
    try:
        pipline = Pipeline()
        pipline.run_pipeline()
    except Exception as e:
        logging.error(f"{e}")
        print(e)


if __name__ == "__main__":
    main()
