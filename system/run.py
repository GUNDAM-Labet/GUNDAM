import argparse
import logging
import os

from utils import ConfigGenerator
from evaluator import config_generation, tune_generation, evaluate_few_shot, save_evaluations

logging.basicConfig(level=logging.INFO)

def run(args):
    train, valid = config_generation(args)
    if args.use_tune:
        train = tune_generation(train, args)
    acc, units, generations, outputs = evaluate_few_shot(train=train, vaild=valid)
    save_evaluations(args=args, acc=acc, num=len(train), units=units, generations=generations, outputs=outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="random seed", default=0, type=int)
    parser.add_argument("--dataset", help="name of dataset", default="cola", type=str)
    parser.add_argument("--src_key", help="source key in jsonl", default="question", type=str)
    parser.add_argument("--tgt_key", help="target key in jsonl", default="answer", type=str)
    parser.add_argument("--from_config", action="store_true", help="load generator from config")
    parser.add_argument("--config_name", help="name of config for generator", type=str)
    parser.add_argument("--embed_model", help="name of embedding method for dense retrieval", default="all-mpnet-base-v2", choices=["all-mpnet-base-v2", "text-embedding-ada-002"])
    
    parser.add_argument("--use_tune", help="tune generator or not", action="store_true")
    parser.add_argument("--num_epoch", help="number of epochs to tune", default=1, type=int)
    parser.add_argument("--split_ratio", help="ratio of training samples in tuning", default=0.8, type=float)

    parser.add_argument("--model_name", help="model name of GPT generator", default="EleutherAI/gpt-neo-1.3B", type=str, choices=["EleutherAI/gpt-neo-1.3B", "gpt2-medium", "gpt2-large"])
    parser.add_argument("--model_path", help="path to generator", default="EleutherAI/gpt-neo-1.3B", type=str, choices=["EleutherAI/gpt-neo-1.3B", "gpt2-medium", "gpt2-large"])
    parser.add_argument("--embed_path", help="path to embeddings", type=str)
    parser.add_argument("--output_path", help="path to store output", type=str)
    parser.add_argument("--batch_size", help="batch_size of generator", default=4, type=int)
    parser.add_argument("--use_fp16", action="store_true", help="use fp16")

    parser.add_argument("--n_shots", help="number of few shots", default=2, type=int)
    parser.add_argument("--priority_level", help="iteration of run miner", default=0, type=int)
    parser.add_argument("--retriever", help="name of retriever", default="ran", type=str, choices=["ran", "hard", "sim", "div"])
    parser.add_argument("--fp16", action="store_true", help="use fp16")
    parser.add_argument("--is_autoreg", help="is generator autoregressive", default=True)

    parser.add_argument("--num_generate", help="number of generations", default=1, type=int)
    parser.add_argument("--add_score", action="store_true", help="add scores to outputs")
    parser.add_argument("--temperature", help="temperature for sampling", default=1.0, type=float)
    parser.add_argument("--decode_method", help="decode method", default="greedy", choices=["greedy", "beam", "sample"])
    parser.add_argument("--num_batch", help="number of batches to generate", default=None, type=int, required=False)
    parser.add_argument("--max_new_tokens", help="max new tokens to generate", default=1, type=int, required=False)
    parser.add_argument("--max_len", help="max length to generate", default=None, type=int, required=False)
    parser.add_argument("--add_io_sep", type=str, default="true", help="add io sep")

    args = parser.parse_args()
    
    current_path = os.path.abspath(os.getcwd())
    args.model_path = os.path.join(os.path.dirname(current_path), "data/") if args.model_path is None else args.model_path
    args.embed_path = os.path.join(os.path.dirname(current_path), "data/") if args.embed_path is None else args.model_path
    args.output_path = os.path.join(os.path.dirname(current_path), "data/") if args.output_path is None else args.output_path

    args.add_io_sep = args.add_io_sep.lower() == "true"
    if args.max_len and not args.max_new_tokens:
        logging.warning(
            "max_new_tokens is not set, using max_len instead. we recommend using max_new_tokens to align with huggingface"
        )
        args.max_new_tokens = args.max_len

    cfg_generator4miner = ConfigGenerator()
    cfg_generator4miner.set(decode_method=args.decode_method, add_score=args.add_score, temperature=args.temperature, num_generate=args.num_generate,
                num_batch=args.num_batch, max_new_tokens=args.max_new_tokens)
    args.cfg_generator4miner = cfg_generator4miner

    cfg_generator4evaluator = ConfigGenerator()
    cfg_generator4evaluator.set(decode_method=args.decode_method, add_score=args.add_score, temperature=args.temperature, num_generate=args.num_generate,
                num_batch=args.num_batch, max_new_tokens=args.max_new_tokens)
    args.cfg_generator4evaluator = cfg_generator4evaluator

    run(args)