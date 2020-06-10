from transformers import DistilBertModel, DistilBertTokenizerFast
from CwnGraph import CwnBase, CwnSense
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from tqdm.autonotebook import tqdm
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
        MSG = f"computing sense vectors of {word}"
        for sense_x in tqdm(senses, desc=MSG):
            examples = sense_x.all_examples()
            ex_vecs = self.compute_example_vecs(examples)
            ex_vecs = ex_vecs.numpy()
            sense_vecs.append(ex_vecs)
            sense_ids += [sense_x.id] * ex_vecs.shape[0]

        knn = self.knn(np.vstack(sense_vecs), sense_ids)
        wsd = WsdBaseline(word, knn)
        return wsd

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
                        for i, xs in enumerate(pos_list) if xs for x in xs))

        # batch_indices: List[int]
        # seq_indices: List[int]
        # These two list are of the same length. Each pair of them index a
        # target occurence in the data.
        batch_indices, seq_indices = list(zip(*indices))


        with torch.no_grad():
            output = self.bert(**input_tensors)


        ## debug use: it can check if batch_indices and seq_indices
        ##   index the right element
        # print(self.tokenizer.decode(input_tensors["input_ids"]
        #         [batch_indices, seq_indices].numpy()))

        # ex_vecs: torch.Tensor
        # ex_vecs contains the contextualized vectors of all targets' occurrence
        # in the inputs
        ex_vecs = output[0][batch_indices, seq_indices]

        return ex_vecs

    def knn(self, vectors, labels):
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(vectors, labels)
        return neigh

    def find_target(self, text):
        toks = text.split("<")
        if len(toks) <= 1:
            return (text.replace("<", "").replace(">", ""), None)
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
    def __init__(self, word, knn):
        srv = WsdService()
        self.word = word
        self.cwn = srv.cwn
        self.knn = knn
        self.tokenizer = srv.tokenizer
        self.model = srv.model

    def senses(self):
        return [CwnSense(sid, self.cwn) for sid in self.knn.classes_]

    def predict(self, text):
        if isinstance(text, str):
            raise ValueError("text should be segmented word list")
        
        pos_list = []
        word_indices = []
        textseq = ""
        for widx, w in enumerate(text):
            if w == self.word:
                pos_list.append(len(textseq))
                word_indices.append(widx)
            textseq += w

        input_tensors = self.tokenizer.encode_plus([textseq],
                        return_tensors="pt")        
        indices = [input_tensors.char_to_token(0, p) for p in pos_list]
        with torch.no_grad():
            output = self.model(**input_tensors)[0]        
        vecs = output[0, list(indices)].numpy()

        try:
            ret = []
            for wi, wx in enumerate(word_indices):
                pred_id = self.knn.predict(vecs)[wi]
                ret.append((wx, CwnSense(pred_id, self.cwn)))
            return ret

        except Exception as ex:
            print(ex)
            return None
        
        


