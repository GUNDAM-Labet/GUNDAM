from typing import List, Dict, Type
from utils import Unit


class BaseConverter():
    INTRO = ""
    OUTPUT2LABEL = {}
    
    def unit2code(self, units: List[Unit], target: Unit) -> str:
        res = self.INTRO + "\n" # + sep_token
        for unit in units:
            res += f"source: {unit.source_input}" + "\n"
            if unit.target_output in self.OUTPUT2LABEL:
                res += f"target: {self.OUTPUT2LABEL[unit.target_output]}" + "\n"
            else:
                res += f"target: {unit.target_output}" + "\n"
        res += f"source: {target.source_input}" + "\n"
        res += f"target: "
        return res

    def code2answer(self, code: str) -> str:
        lines = code.strip().split("\n")
        targets = [line for line in lines if line.startswith("target")]
        res = targets[-1].replace("target:", "").strip()
        return res

ConverterList: Dict[str, Type[BaseConverter]] = {}

def register_converter(name: str):
    def wrapper(cls):
        if not issubclass(cls, BaseConverter):
            raise ValueError('all converters must inherit from BaseConverter class')
        ConverterList[name] = cls
        return cls
    return wrapper

def load_converter(name: str) -> BaseConverter:
    if name not in ConverterList:
        raise ValueError(f'converter {name} not registered')
    return ConverterList[name]()

@register_converter("sst2")
class SST2Converter(BaseConverter):
    INTRO = "analyze the sentiment of the following text excerpts, categorizing them as either 'positive', or 'negative'." + "\n"
    OUTPUT2LABEL = {0: "negative", 1: "positive"}

@register_converter("sst5")
class SST5Converter(BaseConverter):
    INTRO = "analyze the sentiment of the following text excerpts, categorizing them as either 'positive', or 'negative'." + "\n"
    OUTPUT2LABEL = {0: "negative", 1: "positive"}

@register_converter("fpb")
class FPBConverter(BaseConverter):
    INTRO = "analyze the sentiment of the following text excerpts, categorizing them as one label from following choices 'positive', 'neutral' and 'negative'." + "\n"
    OUTPUT2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

@register_converter("cola")
class COLAConverter(BaseConverter):
    INTRO = "analyze the linguistic acceptability of the following text excerpts, categorizing them as either 'positive', or 'negative'." + "\n"
    OUTPUT2LABEL = {0: "negative", 1: "positive"}

@register_converter("trec")
class TRECConverter(BaseConverter):
    INTRO = "analyze the topic of the following text excerpts, categorizing them as one label from following 6 choices 'abbreviation', 'entity', 'description', 'human', 'location' and 'number'." + "\n"
    OUTPUT2LABEL = {0: "abbreviation", 1: "entity", 2: "description", 3: "human", 4: "location", 5: "number"}

@register_converter("subj")
class SUBJConverter(BaseConverter):
    INTRO = "determine whether the following text excerpts is 'subjective' or 'objective'." + "\n"
    OUTPUT2LABEL = {0: "objective", 1: "subjective"}