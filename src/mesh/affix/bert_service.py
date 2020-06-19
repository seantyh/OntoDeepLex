from typing import Union, List, Dict
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizerFast

class BertService:
    def __init__(self):
        model_name = "distilbert-base-multilingual-cased"
        # self.model = DistilBertModel.from_pretrained(model_name)                
        self.bert = DistilBertModel.from_pretrained(model_name)
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
            pad_to_max_length=True, return_tensors="pt")
    
    def decode(self, _id):
        return self.tokenizer.decode(_id)

    def transform(self, input_data: Dict[str, any]):
        outputs = self.bert(**input_data)[0]
        return outputs


    def tokenize(self, text: str):
        return self.tokenizer.encode(text)