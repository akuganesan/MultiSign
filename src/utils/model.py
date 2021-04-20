import os
os.environ['TRANSFORMERS_CACHE'] = 'transformer_cache/'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from torch.nn.utils import weight_norm
from transformers import BertTokenizer, BertModel
# from transformers import AutoTokenizer, AutoModel

BERT_MODELS = {"multi" : {"model_name": "bert-base-multilingual-cased",
                          "tokenizer": "bert-base-multilingual-cased"}, 
               
               "en" :    {"model_name": "bert-base-cased",
                          "tokenizer": "bert-base-cased"},
               
               "ms" :    {"model_name": "DeepPavlov/bert-base-multilingual-cased-sentence",
                          "tokenizer": "DeepPavlov/bert-base-multilingual-cased-sentence"},
               
               "de" :    {"model_name": "bert-base-german-cased",
                          "tokenizer": "bert-base-german-cased"}}

class language_encoder(nn.Module):
    def __init__(self, model_path=None, model_type="multi"):
        super(language_encoder, self).__init__()
        if model_type not in BERT_MODELS:
            error_message = "Invalid BERT model {}. Supported BERT models: {}."
            raise ValueError(error_message.format(model_type,
                                                  ", ".join(list(BERT_MODELS.keys()))))
            
        model_name = BERT_MODELS[model_type]["model_name"]
        tokenizer = BERT_MODELS[model_type]["tokenizer"]
        
        if model_path is None:
            self.model = BertModel.from_pretrained(model_name, output_hidden_states=True)
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer)

        else:
            self.model = AutoModel.from_pretrained(model_path,
                                      output_hidden_states = True,
                                      )
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, 'tokenizer_config.json'))            

    def forward(self, x):
        tokenized = self.tokenizer(x, padding=True, return_tensors="pt")
        output = self.model(**tokenized).pooler_output
        return output


class TemporalAttention(nn.Module):
    """Temporal attention for the joint embeddings. 
       Practically self-attention."""

    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        
        self.scaling_factor = torch.rsqrt(torch.tensor(self.hidden_size, dtype=torch.float))
        self.softmax = nn.Softmax(dim=2) 
        self.neg_inf = torch.tensor(-1e7).cuda()
        
        
    def forward(self, queries, keys, values):    
        q = self.Q(queries)
        k = self.K(keys)
        v = self.V(values)

        batch_size = q.shape[0]
        
        unnormalized_attention = torch.bmm(q, k.permute(0, 2, 1)) * self.scaling_factor
        
        # mask out the attention to the future decoder steps
        dec_seq = queries.shape[1]
        
        mask = torch.tril(torch.ones((batch_size, dec_seq, dec_seq))).cuda()
        unnormalized_attention = unnormalized_attention.masked_fill(
            mask == 0, self.neg_inf)

        attention_weights = self.softmax(unnormalized_attention)
        context = torch.bmm(attention_weights, v) 
        return context, attention_weights


# TODO: Finish spacial attention
class SpacialAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SpacialAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_network = None
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)

        
    def forward(self, queries, keys, values):
        pass
            

class Encoder(nn.Module):
    def __init__(self, num_joints=57, num_dim=2, hidden_size=768, num_layers=1):
        super(Encoder, self).__init__()
        self.num_joints = num_joints
        self.num_dim = num_dim
        self.input_size = self.num_joints*self.num_dim
        self.rnn = nn.GRU(self.input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        
    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        outputs, h_n = self.rnn(x)
        return outputs

class TrajectoryPredictor(nn.Module):
    def __init__(self, pose_size, trajectory_size, hidden_size):
        super(TrajectoryPredictor, self).__init__()
        self.lp = nn.Linear(hidden_size, pose_size)
        self.fc = nn.Linear(pose_size+hidden_size, trajectory_size)


    def forward(self, x):
        pose_vector = self.lp(x)
        trajectory_vector = self.fc(torch.cat((pose_vector, x), dim=-1))
#         mixed_vector = torch.cat((trajectory_vector, pose_vector), dim=-1) # original
        mixed_vector = torch.cat((pose_vector, trajectory_vector), dim=-1)
        return mixed_vector        

class TeacherForcing():
    '''
    Sends True at the start of training, i.e. Use teacher forcing maybe.
    Progressively becomes False by the end of training, start using gt to train
    '''
    def __init__(self, max_epoch):
        self.max_epoch = max_epoch

    def __call__(self, epoch, batch_size=1):
        p = epoch*1./self.max_epoch
        random = torch.rand(batch_size)
#         return (p < random).double()
        return (p < random).float()
        
class DecoderCell(nn.Module):
    def __init__(self, hidden_size, pose_size, trajectory_size, use_h=False, use_tp=False, use_lang=False):
        super(DecoderCell, self).__init__()
        self.use_h = 1 if use_h else 0
        self.use_lang = 1 if use_lang else 0
        self.rnn = nn.GRUCell(input_size=pose_size+trajectory_size+hidden_size*(self.use_h+self.use_lang),
                              hidden_size=hidden_size)
#         self.rnn = nn.GRU(input_size=pose_size+trajectory_size+hidden_size*(self.use_h+self.use_lang),
#                               hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        if use_tp:
            self.tp = TrajectoryPredictor(pose_size=pose_size,
                                    trajectory_size=trajectory_size,
                                    hidden_size=hidden_size)
        else:
            self.tp = nn.Linear(hidden_size, pose_size + trajectory_size)

        if self.use_lang:
            self.lin = nn.Linear(hidden_size+pose_size+trajectory_size, pose_size+trajectory_size)

    def forward(self, x, h):
        if self.use_h:
            x_ = torch.cat([x,h], dim=-1)
        else:
            x_ = x
        h_n = self.rnn(x_, h)
        
        tp_n = self.tp(h_n)
        if self.use_lang:
            y = self.lin(x) + tp_n
        else:
            y = x + tp_n
        return y, h_n
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, pose_size, trajectory_size,
               use_h=False, start_zero=False, use_tp=True,
               use_lang=False, use_attn=False, num_layers=1):
        super(Decoder, self).__init__()
        self.input_size = pose_size + trajectory_size
        self.cell = DecoderCell(hidden_size, pose_size, trajectory_size,
                                use_h=use_h, use_tp=use_tp, use_lang=use_lang)
        ## Hardcoded to reach 0% Teacher forcing in 20 epochs
        self.tf = TeacherForcing(0.05)
        self.start_zero = start_zero
        self.use_lang = use_lang
        self.use_attn = use_attn
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.1)

        self.spacial_attention = nn.ModuleList([
            SpacialAttention(hidden_size=self.hidden_size) for i in range(self.num_layers)])
        
        self.temporal_attention = nn.ModuleList([
            TemporalAttention(hidden_size=self.hidden_size) for i in range(self.num_layers)])
        
        self.attention_mlps = nn.ModuleList([
          nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                        nn.ReLU()) for i in range(self.num_layers)])

    
    def forward(self, h, time_steps, gt, epoch=np.inf, attn=None):
        if self.use_lang:
            lang_z = h
            
        if self.start_zero:
            x = h.new_zeros(h.shape[0], self.input_size)
            x = h.new_tensor(torch.rand(h.shape[0], self.input_size))
        else:
            x = gt[:, 0, :] ## starting point for the decoding 

        Y = []
        H = h.unsqueeze(1)
        for t in range(time_steps):
            if self.use_lang:
                if self.use_attn:  ### calculate temporat attention at each time-step
                    H = self.attention(self.dropout(H))
                x, h = self.cell(torch.cat([x, H[:, -1, :]], dim=-1), h)

            else:
                x, h = self.cell(x, h)
            H = torch.cat((H, h.unsqueeze(1)), dim=1)
            Y.append(x.unsqueeze(1))
            if t > 0:
                mask = self.tf(epoch, h.shape[0]).view(-1, 1).to(x.device)
#                 mask = self.tf(epoch, h.shape[0]).double().view(-1, 1).to(x.device)
                x = mask * gt[:, t-1, :] + (1-mask) * x
        return torch.cat(Y, dim=1)

    def sample(self, h, time_steps, start, attn=None):
        if self.use_lang:
            lang_z = h

        #x = torch.rand(h.shape[0], self.input_size).to(h.device).to(h.dtype)
        x = start ## starting point for the decoding 
        Y = []
        H = h.unsqueeze(1)
        for t in range(time_steps):
            if self.use_lang:
                if self.use_attn:  ### calculate temporal attention at each time-step
                    H = self.attention(self.dropout(H))
                x, h = self.cell(torch.cat([x, H[:, -1, :]], dim=-1), h)
            else:
                x, h = self.cell(x, h)
            H = torch.cat((H, h.unsqueeze(1)), dim=1)
            Y.append(x.unsqueeze(1))
        return torch.cat(Y, dim=1)
 

    def attention(self, h):
        for i in range(self.num_layers):
            temp_h, temp_att_w = self.temporal_attention[i](h, h, h)
            h = temp_h + h
            h = self.attention_mlps[i](h)

        return h
    

class Seq2Seq(nn.Module):
    def __init__(self, hidden_size, pose_size, trajectory_size,
               use_h=False, start_zero=False, use_tp=True,
               use_lang=False, use_attn=False, **kwargs):
        super(Seq2Seq, self).__init__()
        if use_attn: ## use_lang must be true if use_attn is true
            use_lang=True
        ## TODO take root rotation out of Trajectory Predictor
        #pose_size += 4
        #trajectory_size -= 4
        input_size = pose_size + trajectory_size
        self.enc = Encoder(input_size, hidden_size)
        self.dec = Decoder(hidden_size, pose_size, trajectory_size,
                           use_h=use_h, start_zero=start_zero,
                           use_tp=use_tp, use_lang=use_lang,
                           use_attn=use_attn)

    def forward(self, x, train=True, epoch=np.inf, attn=None):
        time_steps = x.shape[1]
        enc_vector = self.enc(x)[:, -1, :]
        dec_vector = self.dec(enc_vector, time_steps, gt=x, epoch=epoch, attn=attn)
        return dec_vector, []
    
class Seq2SeqConditioned9(nn.Module):
    ''' 
    Sentence conditioned pose generation
    if train:
    choose from l2p and p2p
    else:
    l2p
    Seq2SeqKwargs = {hidden_size, 
                   use_h:False,
                   use_lang:False, 
                   use_tp:True,
                   start_zero:False, 
                   s2v:'lstm' or 'bert'}
    *JL2P*
    '''
    def __init__(self, chunks, input_size=300, Seq2SeqKwargs={}, load=None):
        super(Seq2SeqConditioned9, self).__init__()
        self.chunks = chunks
        self.hidden_size = Seq2SeqKwargs['hidden_size']
        self.trajectory_size = Seq2SeqKwargs['trajectory_size']
        self.pose_size = Seq2SeqKwargs['pose_size']
        self.seq2seq = Seq2Seq(**Seq2SeqKwargs)
        if load:
            self.seq2seq.load_state_dict(pkl.load(open(load, 'rb')))
            print('Seq2Seq Model Loaded')
        else:
            print('Seq2Seq Model not found. Initialising randomly')

        ## set requires_grad=False for seq2seq parameters
        #for p in self.seq2seq.parameters():
        #  p.requires_grad = False
        
        self.sentence_enc = language_encoder()
        
#         if Seq2SeqKwargs.get('s2v') == 'lstm':
#             self.sentence_enc = LSTMEncoder(self.hidden_size)
#         elif Seq2SeqKwargs.get('s2v') == 'bert' or Seq2SeqKwargs.get('s2v') is None:
#             self.sentence_enc = BertForSequenceEmbedding(self.hidden_size)
    
    def js_divergence(self, p, q):
        pdb.set_trace()
        m = torch.log((p+q)/2)
        return F.kl_div(m, p, reduce='sum') + F.kl_div(m, q, reduce='sum')


    def forward(self, pose, s2v, train=False, epoch=np.inf):
        pose_enc = self.seq2seq.enc(pose)
        language_z, _ = self.sentence_enc(s2v)
        time_steps = pose.shape[-2]

        if torch.rand(1).item() > 0.5 or not train:
            pose_dec = self.seq2seq.dec(language_z, time_steps, gt=pose)
        else:
            pose_dec = self.seq2seq.dec(pose_enc[:, -1, :], time_steps, gt=pose)
        #internal_losses = [F.mse_loss(pose_enc[:, -1, :], language_z, reduction='mean')]
        internal_losses = []
        return pose_dec, internal_losses

    def sample(self, s2v, time_steps, start):
        language_z, _ = self.sentence_enc(s2v)
        pose_dec = self.seq2seq.dec.sample(language_z, time_steps, start)
        #internal_losses = [F.mse_loss(pose_enc[:, -1, :], language_z, reduction='mean')]
        internal_losses = []
        return pose_dec, internal_losses
    
class ConvBlock(torch.nn.Module):
    '''Convolutional Block from Pix2Pix
    '''
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.activation = activation
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm2d(output_size)

    def forward(self, x):
        if self.activation:
            out = self.conv(self.lrelu(x))
        else:
            out = self.conv(x)

        if self.batch_norm:
            return self.bn(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    '''Deconvolutional Block from Pix2Pix
    '''
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=True, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(output_size)
        self.drop = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU(True)
        self.batch_norm = batch_norm
        self.dropout = dropout

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(self.relu(x)))
        else:
            out = self.deconv(self.relu(x))

        if self.dropout:
            return self.drop(out)
        else:
            return out
        
class Generator(torch.nn.Module):
    '''Generator for Pix2Pix Pose2Video Model
    '''
    def __init__(self, input_dim, num_filter, output_dim):
        super(Generator, self).__init__()

        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
        self.conv5 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv6 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv7 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv8 = ConvBlock(num_filter * 8, num_filter * 8, batch_norm=False)
        # Decoder
        self.deconv1 = DeconvBlock(num_filter * 8, num_filter * 8, dropout=True)
        self.deconv2 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.deconv3 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.deconv4 = DeconvBlock(num_filter * 8 * 2, num_filter * 8)
        self.deconv5 = DeconvBlock(num_filter * 8 * 2, num_filter * 4)
        self.deconv6 = DeconvBlock(num_filter * 4 * 2, num_filter * 2)
        self.deconv7 = DeconvBlock(num_filter * 2 * 2, num_filter)
        self.deconv8 = DeconvBlock(num_filter * 2, output_dim, batch_norm=False)

    def forward(self, x):
        # Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        enc6 = self.conv6(enc5)
        enc7 = self.conv7(enc6)
        enc8 = self.conv8(enc7)
        # Decoder with skip-connections
        dec1 = self.deconv1(enc8)
        dec1 = torch.cat([dec1, enc7], 1)
        dec2 = self.deconv2(dec1)
        dec2 = torch.cat([dec2, enc6], 1)
        dec3 = self.deconv3(dec2)
        dec3 = torch.cat([dec3, enc5], 1)
        dec4 = self.deconv4(dec3)
        dec4 = torch.cat([dec4, enc4], 1)
        dec5 = self.deconv5(dec4)
        dec5 = torch.cat([dec5, enc3], 1)
        dec6 = self.deconv6(dec5)
        dec6 = torch.cat([dec6, enc2], 1)
        dec7 = self.deconv7(dec6)
        dec7 = torch.cat([dec7, enc1], 1)
        dec8 = self.deconv8(dec7)
        out = torch.nn.Tanh()(dec8)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                torch.nn.init.normal(m.deconv.weight, mean, std)
                
class Discriminator(torch.nn.Module):
    '''Discriminator for Pix2Pix Pose2Video Model
    '''
    def __init__(self, input_dim, num_filter, output_dim):
        super(Discriminator, self).__init__()

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8, stride=1)
        self.conv5 = ConvBlock(num_filter * 8, output_dim, stride=1, batch_norm=False)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = torch.nn.Sigmoid()(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)