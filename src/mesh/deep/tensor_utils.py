from typing import List, Union, Dict
import torch
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizerFast
from transformers import DistilBertForMaskedLM
from CwnGraph import CwnSense

class _BertService:
    def __init__(self):
        model_name = "distilbert-base-multilingual-cased"
        # self.model = DistilBertModel.from_pretrained(model_name)
        config = DistilBertConfig.from_pretrained(model_name)
        config.output_attentions = True
        self.mlm = DistilBertForMaskedLM.from_pretrained(model_name, config=config)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    def __promote_to_list(self, text: Union[str, List[str]]) -> List[str]:
        if isinstance(text, str):
            return [text]
        else:
            return text

    def encode(self, text):
        text = self.__promote_to_list(text)
        max_len = max(len(x) for x in text)
        return self.tokenizer.batch_encode_plus(text, max_length=max_len+2,
            pad_to_max_length=True)
    
    def decode(self, _id):
        return self.tokenizer.decode(_id)

    def transform(self, input_data: Dict[str, any], k=5):
        with torch.no_grad():
            predictions = self.mlm(**input_data)
        prob = F.softmax(predictions[0], dim=2)
        logits = predictions[0]
        last_att = predictions[1][-1]

        logits_k, ind_k = torch.topk(logits, k, axis=2)
        logits_k = logits_k.squeeze().numpy()
        ind_k = ind_k.squeeze().numpy()
        return logits_k, ind_k, last_att

    def tokenize(self, text: str):
        return self.tokenizer.encode(text)

class BertService:

    instance = None
    def __init__(self):
        if not BertService.instance:
            BertService.instance = _BertService()
        else:
            pass

    def __getattr__(self, name):
        return getattr(self.instance, name)

def get_cwn_sense_vector(sense: CwnSense):
    pass

