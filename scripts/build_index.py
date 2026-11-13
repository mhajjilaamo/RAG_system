"""
Build a persistent index from config/config.yml.

    python -m scripts.build_index                      # uses config/config.yml
    python -m scripts.build_index --corpus privacy_qa --embedding-model minilm
"""
import argparse

from src.config import load_config
from src.pipeline import build_index


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/config.yml")
    p.add_argument("--corpus", help="override config: contractnli | cuad | maud | privacy_qa")
    p.add_argument("--chunking-strategy", help="override config: naive | recursive")
    p.add_argument("--embedding-model", help="override config: minilm | bge | gemini")
    args = p.parse_args()

    config = load_config(args.config)
    if args.corpus:
        config.corpus = args.corpus
    if args.chunking_strategy:
        config.chunking_strategy = args.chunking_strategy
    if args.embedding_model:
        config.embedding_model = args.embedding_model

    print("index built :", build_index(config))


if __name__ == "__main__":
    main()
