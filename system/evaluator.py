import torch
import numpy as np
import random
import datasets
import logging
import tqdm
import os
import json

from transformers import is_torch_available
from manager import BaseManager, GUNDAMManager
from generator import Generator
from converter import BaseConverter, SentimentConverter
from miner import BaseMiner, Miner
from retriever import BaseRetriever, HardRetriever, RandomRetriever, SimilarRetriever, DiverseRetriever

logging.basicConfig(level=logging.INFO)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def compute_acc(generations, outputs, output2labels):
    num, acc = 0, 0
    for generation, output in zip(generations, outputs):
        if generation == output2labels[output]:
            acc += 1
        num += 1
    return acc / num * 100.0

def config_generation(args):
    set_seed(args.seed)
    dataset = datasets.load_dataset(args.dataset)

    valid = BaseManager(data_type="validation", embed_path=args.embed_path)
    valid.load(data_dict=dataset, source_key=args.src_key, target_key=args.tgt_key)

    train = GUNDAMManager(data_path=args.data_path, data_type="train", embed_path=args.embed_path)
    if args.data_path == "":
        train.load(data_dict=dataset, source_key=args.src_key, target_key=args.src_key)
    else:
        train.load(source_key=args.src_key, target_key=args.src_key)
    generator4miner = Generator(model_name=args.model_name, model_path=args.model_path, from_config=args.from_config, 
        config_name=args.config_name, is_autoreg=args.is_autoreg, batch_size=args.batch_size, use_fp16=args.use_fp16)
    generator4miner.model.zero_grad()
    generator4miner.model.eval()
    logging.info("===== GENERATOR4MINER LOADED =====" + "\n")
    logging.info(f"generator4miner tokenizer length is {len(generator4miner.tokenizer)}" + "\n")
    converter = SentimentConverter()
    train.converter = converter
    miner = Miner(generator=generator4miner, converter=converter)
    train.miner = miner
    retriever = {"hard": HardRetriever, "ran": RandomRetriever, "sim": SimilarRetriever, "div": DiverseRetriever}[args.retriever]
    train.retriever = retriever
    
    generator4evaluator = Generator(model_name=args.model_name, model_path=args.model_path, from_config=args.from_config, 
        config_name=args.config_name, is_autoreg=args.is_autoreg, batch_size=args.batch_size, use_fp16=args.use_fp16)
    train.generator = generator4evaluator
    logging.info("===== GENERATOR4EVALUSTOR LOADED =====" + "\n")
    logging.info(f"generator4evaluator tokenizer length is {len(generator4evaluator.tokenizer)}" + "\n")

    train.check()   # check whether all the components are loaded

    _, max_priority_level = train.get_priority_data()
    while args.priority_level > max_priority_level:
        train.update()  # generate args.priority_level data
    train.set_retriever(priority_level=args.priority_level, n_shots=args.n_shots)
    if args.cfg_generator4evaluator:
        train.set_generator(cfg=args.cfp_generator4evaluator)
    if args.cfg_generator4miner:
        train.set_miner(cfg=args.cfg_generator4miner)
    logging.info("===== GUNDAMManager LOADED ======" + "\n")
    return (train, valid)

def evaluate_unsupervised_generation(train: GUNDAMManager, vaild: BaseManager):
    train.generator.model.zero_grad()
    train.generator.model.eval()

    batch_size = train.generator.batch_size
    units, generations, outputs = [], [], []
    for idx, batch in tqdm(enumerate(vaild.batch()), total=len(vaild)//batch_size):
        generations = train.act(idx, batch)
        units.extend([unit for unit in batch])
        generations.extend([train.converter.code2answer(generation) for generation in generations])
        outputs.extend([unit.target_output for unit in batch])
    
    acc = compute_acc(generations=generations, outputs=outputs, output2labels=train.converter.OUTPUT2LABEL)
    print(f"===== ACC is {acc} =====" + "\n")
    return (acc, units, generations, outputs)
    
def save_evaluations(args, acc, num, units, generations, outputs):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    output_path = os.path.join(args.output_path, f"priority_{args.priority_level}_shots_{args.n_shots}")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    with open(os.path.join(output_path, f"_res.json"), "w") as f:
        f.write(json.dumps({"acc": acc, "num": num}) + "\n") 
    with open(os.path.join(output_path, f"_out.json"), "w") as f:
        for unit, generation, output in zip(units, generations, outputs):
            f.write(json.dumps(
                {"uid": unit.unit_id, "input": unit.source_input, "generation": generation, "output": output}
            ))