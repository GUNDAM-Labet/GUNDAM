import torch

from typing import List, Dict
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoConfig
)
import logging

logging.basicConfig(level=logging.INFO)


def chunks(lst: List, n: int):  # yield successive n-sized chunks from lst
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


class Generator:
    def __init__(self, model_name, model_path, use_config: bool = False, config_name: str = None, 
                 is_autoreg: bool = True, batch_size: int = 32, use_fp16: bool = False):
        self.is_autoreg = is_autoreg
        self.use_fp16 = use_fp16
        if use_config:
            tokenizer, model = self.load_from_config(config_name, model_path)
        else:
            tokenizer, model = self.load_from_pretrained(config_name, model_name, model_path)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"use device: {self.device}, load model from {model_path}")
        self.tokenizer = tokenizer
        self.model = model.to(self.device).eval()

        if self.is_autoreg:
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            
            if len(self.tokenizer) != self.model.config.vocab_size:
                print(f"tokenizer vocab size {len(self.tokenizer)} not match model vocab size {self.model.config.vocab_size}")
                self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.batch_size = batch_size
        if hasattr(self.model.config, "n_positions"):
            self.max_context_len = self.model.config.n_positions
        elif hasattr(self.model.config, "n_ctx"):
            self.max_context_len = self.model.config.n_ctx
        else:
            self.max_context_len = 4096
        

    def load_from_config(self, config_name: str, model_path: str):
        assert self.is_autoreg, "config only works for autoreg now"
        config = AutoConfig.from_pretrained(config_name)
        model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(config_name)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        if self.use_fp16:
            model = model.half()
        return (tokenizer, model)
    
    def load_from_pretrained(self, config_name: str, model_name: str, model_path: str):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except OSError: # for cases where only the model is saved, not the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config_name)
        
        if self.is_autoreg:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            if self.use_fp16:
                model = model.half()
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        return (tokenizer, model)

    def act(self, input_text, decode_method: str = "beam", add_score: bool = False, 
                num_generate: int = 5, num_return_sequence: int = None, temperature: float = 1.0, num_batch: int = None) -> Dict:
        input2output = []

        method_to_kwargs = {
            "beam": {
                "num_beams": num_generate,
                "early_stopping": True,
                "num_return_sequences": num_return_sequence if num_return_sequence else num_generate
            },
            "greedy": {
                "do_sample": False
            },
            "sample": {
                "do_sample": True,
                "num_return_sequences": num_return_sequence if num_return_sequence else num_generate
            },
            "greedy_add_scores": {
                "return_dict_in_gen": True,
                "output_score": True
            }
        }
        common_kwargs = {
            "temperature": temperature
        }
        if add_score and decode_method == "greedy":
            decode_method = "greedy_add_scores"
        kwargs = method_to_kwargs[decode_method]
        kwargs.update(common_kwargs)

        scores = []
        batch_idx = 0
        with torch.no_grad():
            for input_text_batch in list(chunks(input_text, self.batch_size)):
                if num_batch:
                    if batch_idx >= num_batch:
                        break
                
                with torch.cuda.amp.autocast():
                    input_idx = self.tokenizer(
                        input_text_batch,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length"
                    ).to(self.device)

