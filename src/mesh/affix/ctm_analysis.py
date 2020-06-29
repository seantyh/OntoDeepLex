import torch
import torch.nn.functional as F
import pickle
from ..utils import get_data_dir, ensure_dir
from . import ctm_import
import numpy as np
from .affix_ckip import Affixoid

#pylint: disable=import-error
from contextualized_topic_models.models.ctm import CTM

class CtmModel:
    def __init__(self, ctm, vocab):
        self.model = ctm
        self.vocab = vocab
        self._topic_entropy = None

    def get_beta(self):
        return self.model.best_components
    
    def get_topic_entropy(self):
        if self._topic_entropy is None:
            topic_distr = F.softmax(self.get_beta(), dim=0)
            topic_entropy = (-torch.log(topic_distr) * topic_distr).sum(0)
            self._topic_entropy = topic_entropy.detach().cpu().numpy()
        return self._topic_entropy
            
    def get_charloc_entropy(self, charloc: str):
        _id = self.vocab.encode(charloc)
        topic_entropy = self.get_topic_entropy()
        return topic_entropy[_id]
    
    def get_affixoid_entropy(self, affixoid: Affixoid):
        _id = self.vocab.encode(affixoid.affix_form())
        topic_entropy = self.get_topic_entropy()
        return topic_entropy[_id]
    
    def get_topic_list(self, k=10):
        topics = []        
        beta = self.get_beta().detach().cpu().numpy()
        vocab = self.vocab
        for i in range(self.model.n_components):
            idxs = np.argsort(beta[i])[-k:]
            component_words = \
                [vocab.decode(int(idx)) for idx in idxs]
            topics.append(component_words)
        return topics

def get_ctm_models(model_name=None):
    affix_dir = get_data_dir() / "affix"
    model_dir = affix_dir / "ctm_context_models"
    if not model_name:
        model_name = "AVTIM-h100_100-c100-epoch_99.pth"

    with (affix_dir/"affixoid_ctm_vocab.pkl").open("rb") as vocab_f:
        vocab = pickle.load(vocab_f)

    bert_dim = 768
    nvocab = len(vocab)
    ctm = CTM(input_size=nvocab, bert_input_size=bert_dim, inference_type="contextual")
    ctm.load_from_path(model_dir/model_name)
    model = CtmModel(ctm, vocab)

    return model

def get_bychar_ctm_models(model_name=None):
    affix_dir = get_data_dir() / "affix"
    model_dir = affix_dir / "bychar_models"
    if not model_name:
        model_name = "AVTIM-bychar-h100_100-c100-prodLDA-epoch_9.pth"

    with (affix_dir/"bychar_ctm_vocab.pkl").open("rb") as vocab_f:
        vocab = pickle.load(vocab_f)

    bert_dim = 768
    nvocab = len(vocab)
    ctm = CTM(input_size=nvocab, bert_input_size=bert_dim, 
            inference_type="contextual", model_type="LDA")
    ctm.load_from_path(model_dir/model_name)
    model = CtmModel(ctm, vocab)

    return model


