from typing import List
from utils import Unit


class BaseConverter():
    INTRO = ""
    OUTPUT2LABEL = {}
    
    def unit2code(self, units: List[Unit], target: Unit) -> str:
        raise NotImplementedError

    def code2answer(self, code: str) -> str:
        raise NotImplementedError


class SentimentConverter(BaseConverter):
    INTRO = "analyze the sentiment of the following text excerpts, categorizing them as either 'positive', or 'negative'."
    OUTPUT2LABEL = {0: "negative", 1: "positive"}

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