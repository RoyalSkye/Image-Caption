#!/usr/bin/env python3

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import argparse
# import transformer, models


def evaluate_lstm(args):
    """
    Evaluation for decoder_mode: lstm

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    beam_size = args.beam_size
    Caption_End = False
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    with torch.no_grad():
        for i, (image, caps, caplens, allcaps) in enumerate(tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
            k = beam_size
            # Move to GPU device, if available
            image = image.to(device)  # [1, 3, 256, 256]

            # Encode
            encoder_out = encoder(image)  # [1, enc_image_size=14, enc_image_size=14, encoder_dim=2048]
            enc_image_size = encoder_out.size(1)
            encoder_dim = encoder_out.size(-1)
            # # Flatten encoding
            encoder_out = encoder_out.view(1, -1, encoder_dim)  # [1, num_pixels=196, encoder_dim=2048]
            num_pixels = encoder_out.size(1)
            # We'll treat the problem as having a batch size of k, where k is beam_size
            encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # [k, enc_image_size, enc_image_size, encoder_dim]
            # Tensor to store top k previous words at each step; now they're just <start>
            k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # [k, 1]

            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words
            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(device)
            # Lists to store completed sequences and scores
            complete_seqs = []
            complete_seqs_scores = []

            # Start decoding
            step = 1
            h, c = decoder.init_hidden_state(encoder_out)
            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:
                embeddings = decoder.embedding(k_prev_words).squeeze(1)  # [s, embed_dim]
                awe, _ = decoder.attention(encoder_out, h)  # attention_weighted_encoding: [s, encoder_dim], [s, num_pixels]
                gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe
                h, c = decoder.lstm(torch.cat([embeddings, awe], dim=1), (h, c))  # [s, decoder_dim]
                scores = decoder.fc(h)  # [s, vocab_size]
                scores = F.log_softmax(scores, dim=1)
                # top_k_scores: [s, 1]
                scores = top_k_scores.expand_as(scores) + scores  # [s, vocab_size]
                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words // vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)
                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                # Set aside complete sequences
                if len(complete_inds) > 0:
                    Caption_End = True
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly
                # Proceed with incomplete sequences
                if k == 0:
                    break

                seqs = seqs[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
                # Break if things have been going on too long
                if step > 50:
                    break
                step += 1

            # choose the caption which has the best_score.
            assert Caption_End
            indices = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[indices]
            # References
            img_caps = allcaps[0].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)
            # Hypotheses
            hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
            assert len(references) == len(hypotheses)

    # Calculate BLEU1~4, METEOR, ROUGE_L, CIDEr scores
    metrics = get_eval_score(references, hypotheses)

    return metrics


def evaluate_transformer(args):
    """
    Evaluation for decoder_mode: transformer

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    beam_size = args.beam_size
    Caption_End = False
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    with torch.no_grad():
        for i, (image, caps, caplens, allcaps) in enumerate(tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
            k = beam_size
            # Move to GPU device, if available
            image = image.to(device)  # [1, 3, 256, 256]

            # Encode
            encoder_out = encoder(image)  # [1, enc_image_size=14, enc_image_size=14, encoder_dim=2048]
            enc_image_size = encoder_out.size(1)
            encoder_dim = encoder_out.size(-1)
            # We'll treat the problem as having a batch size of k, where k is beam_size
            encoder_out = encoder_out.expand(k, enc_image_size, enc_image_size, encoder_dim)  # [k, enc_image_size, enc_image_size, encoder_dim]
            # Tensor to store top k previous words at each step; now they're just <start>
            # Important: [1, 52] (eg: [[<start> <start> <start> ...]]) will not work, since it contains the position encoding
            k_prev_words = torch.LongTensor([[word_map['<start>']]*52] * k).to(device)  # (k, 52)
            # Tensor to store top k sequences; now they're just <start>
            seqs = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(device)
            # Lists to store completed sequences and scores
            complete_seqs = []
            complete_seqs_scores = []
            step = 1

            # Start decoding
            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:
                # print("steps {} k_prev_words: {}".format(step, k_prev_words))
                # cap_len = torch.LongTensor([52]).repeat(k, 1).to(device) may cause different sorted results on GPU/CPU in transformer.py
                cap_len = torch.LongTensor([52]).repeat(k, 1)  # [s, 1]
                scores, _, _, _, _ = decoder(encoder_out, k_prev_words, cap_len)
                scores = scores[:, step-1, :].squeeze(1)  # [s, 1, vocab_size] -> [s, vocab_size]
                scores = F.log_softmax(scores, dim=1)
                # top_k_scores: [s, 1]
                scores = top_k_scores.expand_as(scores) + scores  # [s, vocab_size]
                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words // vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                # Set aside complete sequences
                if len(complete_inds) > 0:
                    Caption_End = True
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly
                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                # Important: this will not work, since decoder has self-attention
                # k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1).repeat(k, 52)
                k_prev_words = k_prev_words[incomplete_inds]
                k_prev_words[:, :step+1] = seqs  # [s, 52]
                # k_prev_words[:, step] = next_word_inds[incomplete_inds]  # [s, 52]
                # Break if things have been going on too long
                if step > 50:
                    break
                step += 1

            # choose the caption which has the best_score.
            assert Caption_End
            indices = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[indices]
            # References
            img_caps = allcaps[0].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)
            # Hypotheses
            # tmp_hyp = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
            assert len(references) == len(hypotheses)
            # Print References, Hypotheses and metrics every step
            # words = []
            # # print('*' * 10 + 'ImageCaptions' + '*' * 10, len(img_captions))
            # for seq in img_captions:
            #     words.append([rev_word_map[ind] for ind in seq])
            # for i, seq in enumerate(words):
            #     print('Reference{}: '.format(i), seq)
            # print('Hypotheses: ', [rev_word_map[ind] for ind in tmp_hyp])
            # metrics = get_eval_score([img_captions], [tmp_hyp])
            # print("{} - beam size {}: BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
            #       (args.decoder_mode, args.beam_size, metrics["Bleu_1"], metrics["Bleu_2"], metrics["Bleu_3"],
            #        metrics["Bleu_4"],
            #        metrics["METEOR"], metrics["ROUGE_L"], metrics["CIDEr"]))

    # Calculate BLEU1~4, METEOR, ROUGE_L, CIDEr scores
    metrics = get_eval_score(references, hypotheses)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Captioning')
    parser.add_argument('--data_folder', default="./dataset/generated_data",
                        help='folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default="coco_5_cap_per_img_5_min_word_freq",
                        help='base name shared by data files.')
    parser.add_argument('--decoder_mode', default="transformer", help='which model does decoder use?')  # lstm or transformer
    parser.add_argument('--beam_size', type=int, default=3, help='beam_size.')
    parser.add_argument('--checkpoint', default="./BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar",
                        help='model checkpoint.')
    args = parser.parse_args()

    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # transformer.device = torch.device("cpu")
    # models.device = torch.device("cpu")
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    print(device)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    # print(encoder)
    # print(decoder)

    # Load word map (word2id)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    vocab_size = len(word_map)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Normalization transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.decoder_mode == "lstm":
        metrics = evaluate_lstm(args)
    elif args.decoder_mode == "transformer":
        metrics = evaluate_transformer(args)

    print("{} - beam size {}: BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
          (args.decoder_mode, args.beam_size, metrics["Bleu_1"],  metrics["Bleu_2"],  metrics["Bleu_3"],  metrics["Bleu_4"],
           metrics["METEOR"], metrics["ROUGE_L"], metrics["CIDEr"]))
