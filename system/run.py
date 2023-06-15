import argparse
import logging

from config import ConfigGenerator
from evaluator import config_generation, evaluate_unsupervised_generation, save_evaluations

logging.basicConfig(level=logging.INFO)

def run_unsupervised_generation(args):
    train, valid = config_generation(args)
    acc, units, generations, outputs = evaluate_unsupervised_generation(train=train, vaild=valid)
    save_evaluations(acc=acc, num=len(train), units=units, generations=generations, outputs=outputs)

def run_supervised_generation(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="random seed", default=0, type=int)
    parser.add_argument("--model_path", help="path to generator", type=str, required=True)
    parser.add_argument("--embed_path", help="path to embeddings", type=str, required=True)
    parser.add_argument("--output_path", help="path to store output", type=str, required=True)
    parser.add_argument("--dataset", help="name of dataset", default="sst-2", type=str)
    parser.add_argument("--src_key", help="source key in jsonl", default="question", type=str)
    parser.add_argument("--tgt_key", help="target key in jsonl", default="answer", type=str)
    parser.add_argument("--from_config", action="store_true", help="load generator from config")
    parser.add_argument("--config_name", help="name of config for generator", type=str)
    parser.add_argument("--mode", help="mode for evaluation", default="unsupervised", type=str, choices=["unsupervised", "supervised"])

    parser.add_argument("--n_shots", help="number of few shots", default=2, type=int)
    parser.add_argument("--priority_level", help="iteration of run miner", default=1, type=int)
    parser.add_argument("--retriever", help="name of retriever", default="ran", type=str, choices=["ran", "hard", "sim", "div"])
    parser.add_argument("--fp16", action="store_true", help="use fp16")
    parser.add_argument("--is_autoreg", help="is generator autoregressive", default=True)

    parser.add_argument("--num_generate", help="number of generations", default=1, type=int)
    parser.add_argument("--add_score", action="store_true", help="add scores to outputs")
    parser.add_argument("--temperature", help="temperature for sampling", default=1.0, type=float)
    parser.add_argument("--decode_method", help="decode method", default="greedy", choices=["greedy", "beam", "sample"])
    parser.add_argument("--num_batch", help="number of batches to generate", default=None, type=int, required=False)
    parser.add_argument("--max_new_tokens", help="max new tokens to generate", default=None, type=int, required=False)
    parser.add_argument("--max_len", help="max length to generate", default=None, type=int, required=False)

    parser.add_argument("--add_io_sep", type=str, default="true", help="add io sep")

    args = parser.parse_args()
    args.add_io_sep = args.add_io_sep.lower() == "true"
    if args.max_len and not args.max_new_tokens:
        logging.warning(
            "max_new_tokens is not set, using max_len instead. we recommend using max_new_tokens to align with huggingface"
        )
        args.max_new_tokens = args.max_len

    cfg = ConfigGenerator()
    cfg.set(decode_method=args.decode_method, add_score=args.add_score, temperature=args.temperature, num_generate=args.num_generate,
                num_batch=args.num_batch, max_new_tokens=args.max_new_tokens)
    args.cfg_generator4miner = cfg
    args.cfg_generator4evaluator = cfg
    
    if args.mode == "unsupervised":
        run_unsupervised_generation(args)
    elif args.mode == "supervised":
        run_supervised_generation(args)
    else:
        raise NotImplementedError