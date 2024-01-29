import torch
import numpy as np
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
from torch.nn.functional import softmax


class GPT2():
    def __init__(self, path, device='cpu'):
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(path).eval().to(self.device)
        self.tokenize = GPT2Tokenizer.from_pretrained(path)


    def encode(self, words):
        """map from words to ids
        """
        return [self.tokenize.encode(x) for x in words]
        pass


    def get_story_array(self, words, context_words):
        """get word ids for each phrase in a stimulus story
        """
        nctx = context_words + 1
        story_ids = self.encode(words)
        words_token_len = [len(x) for x in story_ids]
        story_ids_expanded = []
        for i in range(len(story_ids)):
            story_ids_expanded.extend(story_ids[i])
        story_ids = story_ids_expanded
        story_array = np.zeros([len(story_ids), nctx]) + self.tokenize.eos_token_id
        for i in range(len(story_array)):
            segment = story_ids[i:i + nctx]
            story_array[i, :len(segment)] = segment
        return torch.tensor(story_array).long(), words_token_len

    def get_context_array(self, contexts):
        """get word ids for each context
        """
        context_array = np.array([self.encode(words) for words in contexts])
        return torch.tensor(context_array).long()

    def get_hidden(self, ids, layer):
        """get hidden layer representations
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids=ids.to(self.device),
                                 attention_mask=mask.to(self.device), output_hidden_states=True)
        return outputs.hidden_states[layer]

    def get_probs(self, ids):
        """get next word probability distributions
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device))
        probs = softmax(outputs.logits, dim=2).detach().cpu().numpy()
        return probs



class GPT():
    def __init__(self, path, vocab, device = 'cpu'): 
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(path).eval().to(self.device)
        self.vocab = vocab
        self.word2id = {w : i for i, w in enumerate(self.vocab)}
        self.UNK_ID = self.word2id['<unk>']

    def encode(self, words):
        """map from words to ids
        """
        return [self.word2id[x] if x in self.word2id else self.UNK_ID for x in words]
        
    def get_story_array(self, words, context_words):
        """get word ids for each phrase in a stimulus story
        """
        nctx = context_words + 1
        story_ids = self.encode(words)
        story_array = np.zeros([len(story_ids), nctx]) + self.UNK_ID
        for i in range(len(story_array)):
            segment = story_ids[i:i+nctx]
            story_array[i, :len(segment)] = segment
        return torch.tensor(story_array).long()

    def get_context_array(self, contexts):
        """get word ids for each context
        """
        context_array = np.array([self.encode(words) for words in contexts])
        return torch.tensor(context_array).long()

    def get_hidden(self, ids, layer):
        """get hidden layer representations
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids = ids.to(self.device), 
                                 attention_mask = mask.to(self.device), output_hidden_states = True)
        return outputs.hidden_states[layer]

    def get_probs(self, ids):
        """get next word probability distributions
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids = ids.to(self.device), attention_mask = mask.to(self.device))
        probs = softmax(outputs.logits, dim = 2).detach().cpu().numpy()
        return probs