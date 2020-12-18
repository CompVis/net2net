"""Code is based on https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning"""

import os, sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from net2net.ckpt_util import get_ckpt_path

#import warnings
#warnings.filterwarnings("ignore")

from net2net.modules.captions.models import Encoder, DecoderWithAttention


rescale = lambda x: 0.5*(x+1)


def imresize(img, size):
    return np.array(Image.fromarray(img).resize(size))


class Img2Text(nn.Module):
    def __init__(self):
        super().__init__()
        model_path = get_ckpt_path("coco_captioner", "net2net/modules/captions")
        word_map_path = "data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"

        # Load word map (word2ix)
        with open(word_map_path, 'r') as j:
            word_map = json.load(j)
        rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
        self.word_map = word_map
        self.rev_word_map = rev_word_map

        checkpoint = torch.load(model_path)

        self.encoder = Encoder()
        self.decoder = DecoderWithAttention(embed_dim=512, decoder_dim=512, attention_dim=512, vocab_size=9490)
        missing, unexpected = self.load_state_dict(checkpoint, strict=False)
        if len(missing) > 0:
            print(f"Missing keys in state-dict: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys in state-dict: {unexpected}")
        self.encoder.eval()
        self.decoder.eval()

        resize = transforms.Lambda(lambda image: F.interpolate(image, size=(256, 256), mode="bilinear"))
        normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        norm = torchvision.transforms.Lambda(lambda image: torch.stack([normalize(rescale(x)) for x in image]))
        self.img_transform = transforms.Compose([resize, norm])
        self.device = "cuda"

    def _pre_process(self, x):
        x = self.img_transform(x)
        return x

    @property
    def mean(self):
        return [0.485, 0.456, 0.406]

    @property
    def std(self):
        return [0.229, 0.224, 0.225]

    def forward(self, x):
        captions = list()
        for subx in x:
            subx = subx.unsqueeze(0)
            captions.append(self.make_single_caption(subx))
        return captions

    def make_single_caption(self, x):
        seq = self.caption_image_beam_search(x)[0][0]
        words = [self.rev_word_map[ind] for ind in seq]
        words = words[:50]
        #if len(words) > 50:
        #    return np.array(['<toolong>'])
        text = ''
        for word in words:
            text += word + ' '
        return text

    def caption_image_beam_search(self, image, beam_size=3):
        """
        Reads a batch of images and captions each of it with beam search.
        :param image: batch of pytorch images
        :param beam_size: number of sequences to consider at each decode-step
        :return: caption, weights for visualization
        """

        k = beam_size
        vocab_size = len(self.word_map)

        # Encode
        # image is a batch of images
        encoder_out_ = self.encoder(image)  # (b, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out_.size(1)
        encoder_dim = encoder_out_.size(3)
        batch_size = encoder_out_.size(0)

        # Flatten encoding
        encoder_out_ = encoder_out_.view(batch_size, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out_.size(1)

        sequences = list()
        alphas_ = list()
        # We'll treat the problem as having a batch size of k per example
        for single_example in encoder_out_:
            single_example = single_example[None, ...]
            encoder_out = single_example.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

            # Tensor to store top k previous words at each step; now they're just <start>
            k_prev_words = torch.LongTensor([[self.word_map['<start>']]] * k).to(self.device)  # (k, 1)

            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)

            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(self.device)  # (k, 1)

            # Tensor to store top k sequences' alphas; now they're just 1s
            seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(self.device)  # (k, 1, enc_image_size, enc_image_size)

            # Lists to store completed sequences, their alphas and scores
            complete_seqs = list()
            complete_seqs_alpha = list()
            complete_seqs_scores = list()

            # Start decoding
            step = 1
            h, c = self.decoder.init_hidden_state(encoder_out)

            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:
                embeddings = self.decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
                awe, alpha = self.decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
                alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
                gate = self.decoder.sigmoid(self.decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe
                h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
                scores = self.decoder.fc(h)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)
                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words // vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences, alphas
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                       dim=1)  # (s, step+1, enc_image_size, enc_image_size)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != self.word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                seqs_alpha = seqs_alpha[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                # Break if things have been going on too long
                if step > 50:
                    break
                step += 1

            try:
                i = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[i]
                alphas = complete_seqs_alpha[i]
            except ValueError:
                print("Catching an empty sequence.")
                try:
                    len_ = len(sequences[-1])
                    seq = [0]*len_
                    alphas = None
                except:
                    seq = [0]*9
                    alphas = None

            sequences.append(seq)
            alphas_.append(alphas)

        return sequences, alphas_

    def visualize_text(self, root, images, sequences, n_row=5, img_name='examples'):
        """
        plot the text corresponding to the given images in a matplotlib figure.
        images are a batch of pytorch images
        """

        n_img = images.size(0)
        n_col = max(n_img // n_row + 1, 2)

        fig, ax = plt.subplots(n_row, n_col)

        i = 0
        j = 0
        for image, seq in zip(images, sequences):
            if i == n_row:
                i = 0
                j += 1
            image = image.cpu().numpy().transpose(1, 2, 0)
            image = 255*(0.5*(image+1))
            image = Image.fromarray(image.astype('uint8'))
            image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
            words = [self.rev_word_map[ind] for ind in seq]
            if len(words) > 50:
                return
            text = ''
            for word in words:
                text += word + ' '

            ax[i, j].text(0, 1, '%s' % (text), color='black', backgroundcolor='white', fontsize=12)
            ax[i, j].imshow(image)
            ax[i, j].axis('off')

        plt.savefig(os.path.join(root, img_name + '.png'))


if __name__ == '__main__':
    model = Img2Text()
    print("done.")
