from tqdm import tqdm
from typing import List, Dict, Iterator
from converter import BaseConverter
from generator import BaseGenerator
from utils import Unit

class BaseMiner():
    def __init__(self, generator: BaseGenerator, converter: BaseConverter):
        self.generator = generator
        self.converter = converter
            
    def _batch_data(self, data: Dict[str, Unit], num_check=2) -> Iterator[List[List[Unit]]]:
        batch_size = self.generator.batch_size

        batch = []
        check = [] # num_check = 2 means: INTRO + x1 + y1 + x2 -> y2
        for unit in data.values():
            check.append(unit)
            if len(check) == num_check:
                batch.append(check)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
                check = []
        if check:
            batch.append(check)
        if batch:
            yield batch
    
    def mine(self, data: Dict[str, Unit]):
        raise NotImplementedError


class One2OneMiner(BaseMiner):
    def __init__(self, converter: BaseConverter = None, generator: BaseGenerator = None):
        super().__init__(generator=generator, converter=converter)
    
    def _generate_batch_data(self, data: Dict[str, Unit]):
        all_generations, all_outputs, all_units = [], [], []
        batch_size = self.generator.batch_size
        for batch in tqdm(self._batch_data(data=data, num_check=2), total=len(data)//batch_size):
            if len(batch[-1]) < 2:
                break
            inputs = [self.converter.unit2code(units=[group[0]], target=group[1]) for group in batch]
            generations = self.generator.act(input_text=inputs)
            print("===== DEBUG =====")
            print(generations)
            
            all_generations.extend([self.converter.code2answer(generation) for generation in generations])
            all_outputs.extend([group[1].target_output for group in batch])
            all_units.extend([group[1] for group in batch])
        return (all_generations, all_outputs, all_units)

    def mine(self, data: Dict[str, Unit]):
        output2label = self.converter.OUTPUT2LABEL
        generations, outputs, units = self._generate_batch_data(data=data)
        res = []
        for generation, output, unit in zip(generations, outputs, units):
            if generation != output2label[output]: # not equal -> priority += 1, equal -> priority = priority
                res.append(unit.unit_id)
        return res


class Pair2OneMiner(BaseMiner):
    def __init__(self, generator: BaseGenerator, converter: BaseConverter):
        super().__init__(generator=generator, converter=converter)
        raise NotImplementedError


class One2PairMiner(BaseMiner):
    def __init__(self, generator: BaseGenerator, converter: BaseConverter):
        super().__init__(generator=generator, converter=converter)
        raise NotImplementedError


class Pair2PairMiner(BaseMiner):
    def __init__(self, generator: BaseGenerator, converter: BaseConverter):
        super().__init__(generator=generator, converter=converter)
        raise NotImplementedError