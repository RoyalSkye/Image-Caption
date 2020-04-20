import torch
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
channel_number = 512


class ScaledDotProductAttention(nn.Module):
    def __init__(self, QKVdim):
        super(ScaledDotProductAttention, self).__init__()
        self.QKVdim = QKVdim

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size, n_heads, -1(len_q), QKVdim]
        :param K, V: [batch_size, n_heads, -1(len_k=len_v), QKVdim]
        :param attn_mask: [batch_size, n_heads, len_q, len_k]
        """
        # scores: [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.QKVdim)
        # Fills elements of self tensor with value where mask is True.
        scores.to(device).masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V).to(device)  # [batch_size, n_heads, len_q, QKVdim]
        return context, attn


class Multi_Head_Attention(nn.Module):
    def __init__(self, Q_dim, K_dim, QKVdim, n_heads=8, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.W_Q = nn.Linear(Q_dim, QKVdim * n_heads).to(device)
        self.W_K = nn.Linear(K_dim, QKVdim * n_heads).to(device)
        self.W_V = nn.Linear(K_dim, QKVdim * n_heads).to(device)
        self.n_heads = n_heads
        self.QKVdim = QKVdim
        self.embed_dim = Q_dim
        self.dropout = nn.Dropout(p=dropout)
        self.W_O = nn.Linear(self.n_heads * self.QKVdim, self.embed_dim).to(device)

    def forward(self, Q, K, V, attn_mask):
        """
        In self-encoder attention:
                Q = K = V: [batch_size, num_pixels=196, encoder_dim=2048]
                attn_mask: [batch_size, len_q=196, len_k=196]
        In self-decoder attention:
                Q = K = V: [batch_size, max_len=52, embed_dim=512]
                attn_mask: [batch_size, len_q=52, len_k=52]
        encoder-decoder attention:
                Q: [batch_size, 52, 512] from decoder
                K, V: [batch_size, 196, 2048] from encoder
                attn_mask: [batch_size, len_q=52, len_k=196]
        return _, attn: [batch_size, n_heads, len_q, len_k]
        """
        residual, batch_size = Q, Q.size(0)
        # q_s: [batch_size, n_heads=8, len_q, QKVdim] k_s/v_s: [batch_size, n_heads=8, len_k, QKVdim]
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        # attn_mask: [batch_size, self.n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # attn: [batch_size, n_heads, len_q, len_k]
        # context: [batch_size, n_heads, len_q, QKVdim]
        context, attn = ScaledDotProductAttention(self.QKVdim)(q_s, k_s, v_s, attn_mask)
        # context: [batch_size, n_heads, len_q, QKVdim] -> [batch_size, len_q, n_heads * QKVdim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.QKVdim).to(device)
        # output: [batch_size, len_q, embed_dim]
        output = self.W_O(context)
        output = self.dropout(output)
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout):
        super(PoswiseFeedForwardNet, self).__init__()
        """
        Two fc layers can also be described by two cnn with kernel_size=1.
        """
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=d_ff, kernel_size=1).to(device)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=embed_dim, kernel_size=1).to(device)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim

    def forward(self, inputs):
        """
        encoder: inputs: [batch_size, len_q=196, embed_dim=2048]
        decoder: inputs: [batch_size, max_len=52, embed_dim=512]
        """
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual)


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, dropout, attention_method, n_heads):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=embed_dim, QKVdim=64, n_heads=n_heads, dropout=dropout)
        if attention_method == "ByPixel":
            self.dec_enc_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=2048, QKVdim=64, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=embed_dim, d_ff=2048, dropout=dropout)
        elif attention_method == "ByChannel":
            self.dec_enc_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=196, QKVdim=64, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=embed_dim, d_ff=2048, dropout=dropout)  # need to change

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs: [batch_size, max_len=52, embed_dim=512]
        :param enc_outputs: [batch_size, num_pixels=196, 2048]
        :param dec_self_attn_mask: [batch_size, 52, 52]
        :param dec_enc_attn_mask: [batch_size, 52, 196]
        """
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, n_layers, vocab_size, embed_dim, dropout, attention_method, n_heads):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.tgt_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding.from_pretrained(self.get_position_embedding_table(embed_dim), freeze=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, dropout, attention_method, n_heads) for _ in range(n_layers)])
        self.projection = nn.Linear(embed_dim, vocab_size, bias=False).to(device)
        self.attention_method = attention_method

    def get_position_embedding_table(self, embed_dim):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / embed_dim)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_idx) for hid_idx in range(embed_dim)]

        embedding_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(52)])
        embedding_table[:, 0::2] = np.sin(embedding_table[:, 0::2])  # dim 2i
        embedding_table[:, 1::2] = np.cos(embedding_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(embedding_table).to(device)

    def get_attn_pad_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # In wordmap, <pad>:0
        # pad_attn_mask: [batch_size, 1, len_k], one is masking
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    def get_attn_subsequent_mask(self, seq):
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        subsequent_mask = torch.from_numpy(subsequent_mask).byte().to(device)
        return subsequent_mask

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        :param encoder_out: [batch_size, num_pixels=196, 2048]
        :param encoded_captions: [batch_size, 52]
        :param caption_lengths: [batch_size, 1]
        """
        batch_size = encoder_out.size(0)
        # Sort input data by decreasing lengths.
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # dec_outputs: [batch_size, max_len=52, embed_dim=512]
        # dec_self_attn_pad_mask: [batch_size, len_q=52, len_k=52], 1 if id=0(<pad>)
        # dec_self_attn_subsequent_mask: [batch_size, 52, 52], Upper triangle of an array with 1.
        # dec_self_attn_mask for self-decoder attention, the position whose val > 0 will be masked.
        # dec_enc_attn_mask for encoder-decoder attention.
        # e.g. 9488, 23, 53, 74, 0, 0  |  dec_self_attn_mask:
        # 0 1 1 1 2 2
        # 0 0 1 1 2 2
        # 0 0 0 1 2 2
        # 0 0 0 0 2 2
        # 0 0 0 0 1 2
        # 0 0 0 0 1 1
        dec_outputs = self.tgt_emb(encoded_captions) + self.pos_emb(torch.LongTensor([list(range(52))]*batch_size).to(device))
        dec_outputs = self.dropout(dec_outputs)
        dec_self_attn_pad_mask = self.get_attn_pad_mask(encoded_captions, encoded_captions)
        dec_self_attn_subsequent_mask = self.get_attn_subsequent_mask(encoded_captions)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        if self.attention_method == "ByPixel":
            dec_enc_attn_mask = (torch.tensor(np.zeros((batch_size, 52, 196))).to(device) == torch.tensor(np.ones((batch_size, 52, 196))).to(device))
        elif self.attention_method == "ByChannel":
            dec_enc_attn_mask = (torch.tensor(np.zeros((batch_size, 52, channel_number))).to(device) == torch.tensor(np.ones((batch_size, 52, channel_number))).to(device))

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # attn: [batch_size, n_heads, len_q, len_k]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, encoder_out, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        predictions = self.projection(dec_outputs)
        return predictions, encoded_captions, decode_lengths, sort_ind, dec_self_attns, dec_enc_attns


class EncoderLayer(nn.Module):
    def __init__(self, dropout, attention_method, n_heads):
        super(EncoderLayer, self).__init__()
        """
        In "Attention is all you need" paper, dk = dv = 64, h = 8, N=6
        """
        if attention_method == "ByPixel":
            self.enc_self_attn = Multi_Head_Attention(Q_dim=2048, K_dim=2048, QKVdim=64, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=2048, d_ff=4096, dropout=dropout)
        elif attention_method == "ByChannel":
            self.enc_self_attn = Multi_Head_Attention(Q_dim=196, K_dim=196, QKVdim=64, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=196, d_ff=512, dropout=dropout)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_inputs: [batch_size, num_pixels=196, 2048]
        :param enc_outputs: [batch_size, len_q=196, d_model=2048]
        :return: attn: [batch_size, n_heads=8, 196, 196]
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, n_layers, dropout, attention_method, n_heads):
        super(Encoder, self).__init__()
        if attention_method == "ByPixel":
            self.pos_emb = nn.Embedding.from_pretrained(self.get_position_embedding_table(), freeze=True)
        # self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([EncoderLayer(dropout, attention_method, n_heads) for _ in range(n_layers)])
        self.attention_method = attention_method

    def get_position_embedding_table(self):
        def cal_angle(position, hid_idx):
            x = position % 14
            y = position // 14
            x_enc = x / np.power(10000, hid_idx / 1024)
            y_enc = y / np.power(10000, hid_idx / 1024)
            return np.sin(x_enc), np.sin(y_enc)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_idx)[0] for hid_idx in range(1024)] + [cal_angle(position, hid_idx)[1] for hid_idx in range(1024)]

        embedding_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(196)])
        return torch.FloatTensor(embedding_table).to(device)

    def forward(self, encoder_out):
        """
        :param encoder_out: [batch_size, num_pixels=196, dmodel=2048]
        """
        batch_size = encoder_out.size(0)
        positions = encoder_out.size(1)
        if self.attention_method=="ByPixel":
            encoder_out = encoder_out + self.pos_emb(torch.LongTensor([list(range(positions))]*batch_size).to(device))
        # encoder_out = self.dropout(encoder_out)
        # enc_self_attn_mask: [batch_size, 196, 196]
        enc_self_attn_mask = (torch.tensor(np.zeros((batch_size, positions, positions))).to(device)
                              == torch.tensor(np.ones((batch_size, positions, positions))).to(device))
        enc_self_attns = []
        for layer in self.layers:
            encoder_out, enc_self_attn = layer(encoder_out, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return encoder_out, enc_self_attns


class Transformer(nn.Module):
    """
    See paper 5.4: "Attention Is All You Need" - https://arxiv.org/abs/1706.03762
    "Apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
    In addition, apply dropout to the sums of the embeddings and the positional encodings in both the encoder
    and decoder stacks." (Now, we dont't apply dropout to the encoder embeddings)
    """
    def __init__(self, vocab_size, embed_dim, encoder_layers, decoder_layers, dropout=0.1, attention_method="ByPixel", n_heads=8):
        super(Transformer, self).__init__()
        self.encoder = Encoder(encoder_layers, dropout, attention_method, n_heads)
        self.decoder = Decoder(decoder_layers, vocab_size, embed_dim, dropout, attention_method, n_heads)
        self.embedding = self.decoder.tgt_emb
        self.attention_method = attention_method

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, enc_inputs, encoded_captions, caption_lengths):
        """
        preprocess: enc_inputs:[batch_size, 14, 14, 2048]/[batch_size, 196, 2048] -> [batch_size, 196, 2048]
        encoded_captions: [batch_size, 52]
        caption_lengths: [batch_size, 1], not used
        The encoder or decoder is composed of a stack of n_layers=6 identical layers.
        One layer in encoder: Multi-head Attention(self-encoder attention) with Norm & Residual
                            + Feed Forward with Norm & Residual
        One layer in decoder: Masked Multi-head Attention(self-decoder attention) with Norm & Residual
                            + Multi-head Attention(encoder-decoder attention) with Norm & Residual
                            + Feed Forward with Norm & Residual
        """
        batch_size = enc_inputs.size(0)
        encoder_dim = enc_inputs.size(-1)
        if self.attention_method == "ByPixel":
            enc_inputs = enc_inputs.view(batch_size, -1, encoder_dim)
        elif self.attention_method == "ByChannel":
            enc_inputs = enc_inputs.view(batch_size, -1, encoder_dim).permute(0, 2, 1)  # (batch_size, 2048, 196)

        encoder_out, enc_self_attns = self.encoder(enc_inputs)
        # encoder_out: [batch_size, 196, 2048]
        predictions, encoded_captions, decode_lengths, sort_ind, dec_self_attns, dec_enc_attns = self.decoder(encoder_out, encoded_captions, caption_lengths)
        alphas = {"enc_self_attns": enc_self_attns, "dec_self_attns": dec_self_attns, "dec_enc_attns": dec_enc_attns}
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
