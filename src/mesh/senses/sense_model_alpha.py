from transformers import DistilBertModel, DistilBertTokenizerFast
from CwnGraph import CwnBase
import numpy as np
import torch

class _WsdService:
    def __init__(self):
        model_name = "distilbert-base-multilingual-cased"
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        self.cwn = CwnBase()

class WsdService:
    instance = None
    def __init__(self):
        if not WsdService.instance:
            WsdService.instance = _WsdService()

    def __getattr__(self, key):
        return getattr(self.instance, key)

class WsdBaselineBuilder:
    def __init__(self):
        srv = WsdService()
        self.cwn = srv.cwn
        self.bert = srv.model
        self.tokenizer = srv.tokenizer
    
    def build(self, word):        
        senses= self.cwn.find_all_senses(word)        
        sense_ids = []
        sense_vecs = []
        for sense_x in senses:
            examples = sense_x.all_examples()
            ex_vecs = self.compute_example_vecs(examples)
            ex_vecs = ex_vecs.numpy()
            sense_vecs.append(ex_vecs)
            sense_ids += [sense_x.id] * ex_vecs.shape[0]
        return sense_ids, np.vstack(sense_vecs)
    
    def compute_example_vecs(self, examples):
        # txt_list: List[str]
        #   the list of example sentences
        # pos_list: List[List[Tuple[Start, End]]]
        #   the target position in each sentences. While most of the 
        #   examples contains only one target, some sentences have multiple of them.
        txt_list, pos_list = list(zip(*(self.find_target(x) for x in examples)))

        
        # input_tensors: Dict[str, torch.Tensor]
        input_tensors = self.tokenizer.batch_encode_plus(list(txt_list), 
                                max_length=max(len(x) for x in txt_list)+2, 
                                pad_to_max_length=True, return_tensors="pt")
                  

        # indices: List[Tuple(BatchIdx, SeqIdx)]
        #   The index format more convenient to use with tensors
        indices = list(((i, input_tensors.char_to_token(i, x[0])) \
                        for i, xs in enumerate(pos_list) for x in xs))    
        
        # batch_indices: List[int]
        # seq_indices: List[int]
        # These two list are of the same length. Each pair of them index a 
        # target occurence in the data.
        batch_indices, seq_indices = list(zip(*indices))

        
        with torch.no_grad():
            output = self.bert(**input_tensors)   
                 

        ## debug use: it can check if batch_indices and seq_indices 
        ##   index the right element
        print(self.tokenizer.decode(input_tensors["input_ids"]
                [batch_indices, seq_indices].numpy()))

        # ex_vecs: torch.Tensor
        # ex_vecs contains the contextualized vectors of all targets' occurrence
        # in the inputs
        ex_vecs = output[0][batch_indices, seq_indices]    
        
        return ex_vecs

    def knn(self, vectors, labels):
        pass

    def find_target(self, text):
        toks = text.split("<")
        if len(toks) <= 1:
            return (text.replace("<").replace(">"), None)
        buf = toks[0]
        pos = []
        for tok in toks[1:]:
            xs = tok.split(">")
            if len(xs) <= 1:
                buf.append(tok)
            pos.append((len(buf), len(buf)+len(xs[0])))
            buf += "".join(xs)
        return buf, pos

class WsdBaseline:
    pass