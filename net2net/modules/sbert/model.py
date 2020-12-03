# check out https://github.com/UKPLab/sentence-transformers,
# list of pretrained models @ https://www.sbert.net/docs/pretrained_models.html

from sentence_transformers import SentenceTransformer
import numpy as np
import torch.nn as nn


class SentenceEmbedder(nn.Module):
    def __init__(self, version='bert-large-nli-stsb-mean-tokens'):
        super().__init__()
        np.set_printoptions(threshold=100)
        # Load Sentence model (based on BERT) from URL
        self.model = SentenceTransformer(version, device="cuda")
        self.model.eval()

    def forward(self, sentences):
        """sentences are expect to be a list of strings, e.g.
            sentences = ['This framework generates embeddings for each input sentence',
                         'Sentences are passed as a list of string.',
                         'The quick brown fox jumps over the lazy dog.'
                         ]
        """
        sentence_embeddings = self.model.encode(sentences, batch_size=len(sentences), show_progress_bar=False,
                                                convert_to_tensor=True)
        return sentence_embeddings.cuda()

    def encode(self, sentences):
        embeddings = self(sentences)
        return embeddings[:,:,None,None]


if __name__ == '__main__':
    model = SentenceEmbedder(version='distilroberta-base-paraphrase-v1')
    sentences = ['This framework generates embeddings for each input sentence',
                 'Sentences are passed as a list of string.',
                 'The quick brown fox jumps over the lazy dog.'
                 ]
    emb = model.encode(sentences)
    print(emb.shape)
    print("done.")
