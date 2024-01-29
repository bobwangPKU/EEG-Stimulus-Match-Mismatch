import numpy as np
import torch
torch.set_default_tensor_type(torch.FloatTensor)

class LMFeatures():
    """class for extracting contextualized features of stimulus words
    """

    def __init__(self, model, layer, context_words):
        self.model, self.layer, self.context_words = model, layer, context_words

    def extend(self, extensions, verbose=False):
        """outputs array of vectors corresponding to the last words of each extension
        """
        contexts = [extension[-(self.context_words + 1):] for extension in extensions]
        if verbose: print(contexts)
        context_array = self.model.get_context_array(contexts)
        embs = self.model.get_hidden(context_array, layer=self.layer)
        return embs[:, len(contexts[0]) - 1]

    def get_hidden(self, extensions, ncontext):
        """get hidden layer representations
        """
        contexts = [extension[-(ncontext + 1):] for extension in extensions]
        context_array = self.model.get_context_array(contexts)
        embs = self.model.get_hidden(context_array, layer=self.layer)

        return embs


    def make_stim(self, words):
        """outputs matrix of features corresponding to the stimulus words
        """
        """outputs matrix of features corresponding to the stimulus words
        """
        context_array = self.model.get_story_array(words, self.context_words)
        batch_size = 32
        embs = []
        for i in range(0, context_array.shape[0], batch_size):
            embs.append(self.model.get_hidden(context_array[i:i + batch_size], layer=self.layer))
        embs = torch.cat(embs, dim=0)
        embs = embs.detach().cpu().numpy()
        return np.vstack([embs[0, :self.context_words],
            embs[:context_array.shape[0] - self.context_words, self.context_words]])

            
class LMFeatures1():
    """class for extracting contextualized features of stimulus words
    """
    def __init__(self, model, layer, context_words):
        self.model, self.layer, self.context_words = model, layer, context_words

    def extend(self, extensions, verbose = False):
        """outputs array of vectors corresponding to the last words of each extension
        """
        contexts = [extension[-(self.context_words+1):] for extension in extensions]
        if verbose: print(contexts)
        context_array = self.model.get_context_array(contexts)
        embs = self.model.get_hidden(context_array, layer = self.layer)
        return embs[:, len(contexts[0]) - 1]

    def get_hidden(self, extensions, ncontext):
        """get hidden layer representations
        """
        contexts = [extension[-(ncontext+1):] for extension in extensions]
        context_array = self.model.get_context_array(contexts)
        embs = self.model.get_hidden(context_array, layer = self.layer)

        return embs

    def make_stim(self, words):
        """outputs matrix of features corresponding to the stimulus words
        """
        context_array, words_token_len = self.model.get_story_array(words, self.context_words)
        batch_size = 32
        embs = []
        for i in range(0, context_array.shape[0], batch_size):
            embs.append(self.model.get_hidden(context_array[i:i+batch_size], layer = self.layer))
        embs = torch.cat(embs, dim=0)
        # embs = self.model.get_hidden(context_array, layer = self.layer)
        embs = embs.detach().cpu().numpy()
        emb_np = np.vstack([embs[0, :self.context_words],
                   embs[:context_array.shape[0] - self.context_words, self.context_words]])
        # get word emb for each word acording to words_token_len
        word_embs = []
        words_token_len.insert(0, 0)
        for i in range(len(words_token_len)-1):
            st = sum(words_token_len[0:i+1])
            ed = sum(words_token_len[0:i+2])
            word_embs.append(emb_np[st:ed, :])
        word_embs = [np.mean(word_emb, axis=0) for word_emb in word_embs]
        return np.array(word_embs)