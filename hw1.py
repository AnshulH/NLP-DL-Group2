from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words # added for detecting English words
import nltk
import re
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import argparse
from matplotlib.pylab import plt
from numpy import arange
from scipy import stats
from matplotlib import pyplot
import numpy as np
from numpy import sqrt
from numpy import argmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import numpy as np
import math

# nltk.download('stopwords')
# nltk.download('words')
# nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

def decode(vocab,corpus):
    
    text = ''
    for i in range(len(corpus)):
        wID = corpus[i]
        text = text + vocab[wID] + ' '
    return(text)

def encode(words,text):
    corpus = []
    tokens = text.split(' ')
    for t in tokens:
        try:
            wID = words[t][0]
        except:
            wID = words['<unk>'][0]
        corpus.append(wID)
    return(corpus)

def read_encode_fnn(file_name,vocab,words,corpus,threshold):
    
    wID = len(vocab)
    
    if threshold > -1:
        with open(file_name,'rt', encoding='utf8') as f:
            for line in f:
                line = line.replace('\n','')
                # Added lower-casing
                line = line.lower()
                
                # Strips out all charcters other than alphanumeric
                line = re.sub('[\W_]+', ' ', line, flags=re.UNICODE)
                
                # Strips out numbers
                line = re.sub('\d+', '', line)
                
                tokens = line.split(' ')
                for t in tokens:
                    try:
                        elem = words[t]
                    except:
                        elem = [wID,0]
                        vocab.append(t)
                        wID = wID + 1
                    elem[1] = elem[1] + 1
                    words[t] = elem

        temp = words
        words = {}
        vocab = []
        wID = 0
        words['<unk>'] = [wID,100]
        vocab.append('<unk>')
        for t in temp:
            if temp[t][1] >= threshold:
                vocab.append(t)
                wID = wID + 1
                words[t] = [wID,temp[t][1]]
            
                    
    with open(file_name,'rt', encoding='utf8') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(' ')
            for t in tokens:
                try:
                    wID = words[t][0]
                except:
                    wID = words['<unk>'][0]
                corpus.append(wID)
                
    return [vocab,words,corpus]

def read_encode(file_name,vocab,words,corpus,threshold):
    
    wID = len(vocab)
    if threshold > -1:
        with open(file_name,'rt') as f:
            for line in f:
                line = line.replace('\n','')
                tokens = line.split(' ')
                for t in tokens:
                    try:
                        elem = words[t]
                    except:
                        elem = [wID,0]
                        vocab.append(t)
                        wID = wID + 1
                    elem[1] = elem[1] + 1
                    words[t] = elem

        temp = words
        words = {}
        vocab = []
        wID = 0
        words['<unk>'] = [wID,100]
        wID = 1
        words['< start_bio >'] = [wID,100]
        wID = 2
        words['< end_bio >'] = [wID,100]
        vocab.append('<unk>')
        vocab.append('< start_bio >')
        vocab.append('< end_bio >')
        for t in temp:
            if temp[t][1] >= threshold:
                vocab.append(t)
                wID = wID + 1
                words[t] = [wID,temp[t][1]]
            
                    
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(' ')
            for t in tokens:
                try:
                    wID = words[t][0]
                except:
                    wID = words['<unk>'][0]
                corpus.append(wID)
                
    return [vocab,words,corpus]

def read_alltext(file_name,batch_size,vocab,words):
  '''
  tokenizes data, removes digits, converts to wIDs, and provides a tensor of batched inputs
  inputs: 
    file_name = a complete file path to a mixed dataset containing text and labels
    sequence_length = the length of the desired input array
  outputs:
    x = a tensor containing wIDs of the input
  '''
  word_dict = words
  wid = []
  real_wid = word_dict.get('[REAL]')[0]
  fake_wid = word_dict.get('[FAKE]')[0]

  with open(file_name) as input:
    all_text = input.readlines()

  for line in all_text:
    line = line.strip()
    line = re.sub("[\d-]", "",line)
    if "< start_bio >" in line:
      try:
        wid.append(1)
      except:
        continue
    elif "< end_bio >" in line:
        wid.append(2)
    elif "[REAL]" in line:
        wid.append(real_wid)
    elif "[FAKE]" in line:
        wid.append(fake_wid)
    else:
      line = [token for token in word_tokenize(line.lower())]
      for token in line:
        word_info = word_dict.get(token)
        if word_info is None:
          wid.append(0) 
        else:
          wid.append(word_info[0])

  x = torch.tensor(np.asarray(wid)) 
  num_batches = x.shape[0] // batch_size 
  x = x[:num_batches * batch_size] 
  x = x.view(batch_size, num_batches)   

  return x,vocab,words

def get_batch(data, seq_len, num_batches, idx):
    src = data[:, idx:idx+seq_len]                   
    target = data[:, idx+1:idx+seq_len+1]             
    return src, target

class FFNN(nn.Module):
    # d_model = embedding dimensions
    def __init__(self, vocab, words,d_model, d_hidden, dropout):
        super().__init__() 
    
        self.vocab = vocab
        self.words = words
        self.vocab_size = len(vocab)
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.dropout = nn.Dropout(dropout)
        self.embeds = nn.Embedding(self.vocab_size, d_model)
        
        # Context size * dimensions for input
        # Hidden layer neurons was more difficult to find information
        self.linear1 = nn.Linear(3 * d_model, 512)
        self.linear2 = nn.Linear(512, self.vocab_size)

    def forward(self, src):
        embeds = self.embeds(src).flatten(1, 2)
        out = F.relu(self.linear1(embeds))
        out = self.dropout(out)
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
                
    def init_weights(self):
        pass

class LSTM(nn.Module):
    def __init__(self,vocab,words,d_model,d_hidden,n_layers,dropout_rate, tie_weights):
        super().__init__()
        
        self.vocab = vocab
        self.words = words
        self.vocab_size = len(self.vocab)
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.embeds = nn.Embedding(self.vocab_size,d_model)
#          {perform other initializations needed for the LSTM}
        self.lstm = nn.LSTM(d_model, d_hidden, n_layers, dropout=dropout_rate, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(d_hidden, self.vocab_size)

        if tie_weights:
          assert d_model == d_hidden, 'cannot tie, check dims'
          self.embeds.weight = self.fc.weight
        self.init_weights()
        
    def forward(self,src,h):
        embeds = self.dropout(self.embeds(src))  
        out, h = self.lstm(embeds, h) 
        out = self.dropout(out)
        predict = self.fc(out) 
        return predict, h
    
    def init_weights(self):
        emb_range = 0.1
        init_range = 1/math.sqrt(self.d_hidden)
        self.embeds.weight.data.uniform_(-emb_range, emb_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        for i in range(self.n_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.d_model,
                    self.d_hidden).uniform_(-init_range, init_range) 
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.d_hidden, 
                    self.d_hidden).uniform_(-init_range, init_range) 
        
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.n_layers, batch_size, self.d_hidden).to(device)
        cell = torch.zeros(self.n_layers, batch_size, self.d_hidden).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
      hidden, cell = hidden
      hidden = hidden.detach()
      cell = cell.detach()
      return hidden,cell

def LSTM_train(model, data, optimizer, criterion, batch_size, seq_len, clip, device):

  epoch_loss = 0
  model.train()

  num_batches = data.shape[-1]
  data = data[:, :num_batches - (num_batches -1) % seq_len]
  num_batches = data.shape[-1]

  hidden = model.init_hidden(batch_size, device)
    
  for idx in range(0, num_batches - 1, seq_len):  # The last batch can't be a src
    #   print('works')
      optimizer.zero_grad()
      hidden = model.detach_hidden(hidden)

      src, target = get_batch(data, seq_len, num_batches, idx)
      src, target = src.to(device), target.to(device)
      batch_size = src.shape[0]
      prediction, hidden = model(src, hidden)               

      prediction = prediction.reshape(batch_size * seq_len, -1)   
      target = target.reshape(-1)
      loss = criterion(prediction, target)
      
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
      optimizer.step()
      epoch_loss += loss.item() * seq_len
    
  return math.exp(epoch_loss / num_batches)

def evaluate(model, data, criterion, batch_size, seq_len, device):

    epoch_loss = 0
    model.eval()
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for idx in range(0, num_batches - 1, seq_len):
            hidden = model.detach_hidden(hidden)
            src, target = get_batch(data, seq_len, num_batches, idx)
            src, target = src.to(device), target.to(device)
            batch_size= src.shape[0]

            prediction, hidden = model(src, hidden)
            prediction = prediction.reshape(batch_size * seq_len, -1)
            target = target.reshape(-1)

            loss = criterion(prediction, target)
            epoch_loss += loss.item() * seq_len
    print('works')
    return math.exp(epoch_loss / num_batches)

def make_tensors(file_name,vocab,words):
  '''
  creates a list of tuples containing a list with the tokenized bio and a fake/real label

  inputs: 
    file_name = a complete file path to a mixed dataset containing text and labels 
  outputs:
    x = a tensor containing wIDs for each bio
    y = a tensor containing the corresponding label for each input
  '''
  
  # stop = set(stopwords.words('english') + list(string.punctuation))
  with open(file_name) as input:
    all_text = input.readlines()

# if anybody wants to do some work on the tokenization section, 
# you should work with the split_bios array, since it contains actual words

# split_bios = [([bio1, tokens1],[REAL]),([bio2, tokens2],[FAKE])]
  split_bios = []
  line_text = []
  seq_status = 0

  # seq_status = 0 marks the start of the bio
  # seq_status = 1 is the main text of the bio
  # seq_status = 2 is the end of the bio
  for line in all_text:
    line = line.strip()
    if "< start_bio >" in line:
      seq_status = 0
      continue
    if "< end_bio >" in line:
      seq_status = 2
      line_text.append(line)
      continue
    if seq_status == 0:
      seq_status = 1
    if seq_status == 1:
      line = re.sub("[\d-]", "",line)
      line = [token for token in word_tokenize(line.lower())]
      if len(line) > 0:
        line_text.extend(line)
    if seq_status == 2 and line !="":
      split_bios.append((line_text,line))
      line_text = []
 
  word_dict = words  
  bio_array = [] 
  y = []

  for bio in split_bios:
    wid = []
    wid.append(1)
    for token in bio[0]:
      word_info = word_dict.get(token)
      if word_info is None:
        wid.append(0)
      else:
        wid.append(word_info[0])
    bio_array.append(np.asarray(wid)) 
    real_wid = word_dict.get('[REAL]')[0]
    fake_wid = word_dict.get('[FAKE]')[0] 
    if real_wid == word_dict.get(bio[1])[0]: 
        y.append(real_wid)
    elif fake_wid == word_dict.get(bio[1])[0]:
        y.append(fake_wid) 
    else:
        print('error')
       
  x = bio_array
  y = y  
  return x,y

def get_input_batch(data, seq_len, idx):
    src = data[:, idx:idx+seq_len]                     
    return src 
 
def LSTM_Classify(input, model, vocab, word_dict, device, seed=None, seq_len=30):
  real_wid = word_dict.get('[REAL]')[0]
  fake_wid = word_dict.get('[FAKE]')[0]
  if seed is not None:
      torch.manual_seed(seed)
  model.eval() 
  input = torch.transpose(input, 0, 1) 
  num_batches = input.shape[-1]  
  input = input[:,(num_batches -1) % seq_len :]
  num_batches = input.shape[-1]  
  batch_size = 1  
  hidden = model.init_hidden(batch_size, device)
  prediction = None
  for idx in range(0, num_batches - 1, seq_len): 
            hidden = model.detach_hidden(hidden)
            src  = get_input_batch(input, seq_len, idx)
            src  = src.to(device)  
            prediction, hidden = model(src, hidden)

  with torch.no_grad():
      if(prediction == None): 
        print(input.size())
        return 124
      prediction = prediction.reshape(batch_size * seq_len, -1)  
      probs = torch.softmax(prediction[-1, :] , dim=-1)   
      real_prediction = probs[real_wid]
      fake_prediction = probs[fake_wid] 
      if real_prediction > fake_prediction:
        prediction = real_wid
      else:
        prediction = fake_wid
   
  return prediction 

def create_context_and_next_words(windows, words):
    all_context = []
    all_next_words = []
    skipped_labels = 0

    for (each_bio, bio_label) in windows:
        # Considering only fake windows for training the model
        if bio_label == 0:
            for context, label in each_bio:
                # print(type(words))
                found_in_words = [word in words for word in context]
                found_in_words.extend([label in words])
                if all(found_in_words):
                    all_context.append([words[word][0] for word in context])
                    all_next_words.append([words[label][0]])
                else:
                    all_context.append([0] * len(context))
                    all_next_words.append([0])
                    
                    skipped_labels += 1

    return torch.LongTensor(all_context), torch.LongTensor(all_next_words)

def create_windows(split_bios, window_size):
    sliding_windows = []
    for bio, label in split_bios:
        bio_without_nums = ''.join([i for i in bio if not i.isdigit()])
        tokens = [token for token in bio_without_nums.split(" ") if token != ""]
        
        ngrams = []
        for i in range(len(tokens) - window_size):
            ngrams.append((
                [tokens[i + j] for j in range(window_size)],
                tokens[i + window_size]
            ))
        
        sliding_windows.append((ngrams, label))
    
    return sliding_windows

def read_bios(file_name, labels=False):
    with open(file_name,'rt', encoding='utf8') as f:
        all_bios = f.readlines()
        
    split_bios = []
    curr_bio = ""
    curr_index = 0
    while curr_index < len(all_bios):
        curr_line = all_bios[curr_index].lower()
        # Strips out all charcters other than alphanumeric
        curr_line = re.sub('[\W_]+', ' ', curr_line, flags=re.UNICODE)
        
        # Strips out numbers
        curr_line = re.sub('\d+', '', curr_line)
        
        curr_line = curr_line.strip()
        
        if curr_line == "start bio":
            # Skips their name
            curr_index += 1
        
        elif curr_line == "end bio":
          if labels:
            curr_index += 2
            if "FAKE" in all_bios[curr_index]:
                label = 0
            else:
                label = 1

            split_bios.append((curr_bio, label))
            curr_bio = ""
          
          else:
            split_bios.append((curr_bio, 0))
            curr_bio = ""
        
        else:
            # Check to ensure not empty space
            if curr_line:
                if curr_bio == "":
                    curr_bio = curr_line
                else:
                    curr_bio += " " +  curr_line
        
        curr_index += 1
        
    
    return split_bios

def make_context_and_true_words_per_bio(bio, words):
  all_context = []
  all_next_words = []

  for context, label in bio:
      found_in_words = [word in words for word in context]
      found_in_words.extend([label in words])
      if all(found_in_words):
          all_context.append([words[word][0] for word in context])
          all_next_words.append([words[label][0]])
      else:
          all_context.append([0] * len(context))
          all_next_words.append([0])

    # Random edge case on line 42422 of the bios...
  if all_context == [] and all_next_words == []:
    all_context.append([0] * 3)
    all_next_words.append([0])

  return torch.LongTensor(all_context), torch.LongTensor(all_next_words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_model', type=int, default=100)
    parser.add_argument('-d_hidden', type=int, default=100)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-seq_len', type=int, default=30)
    parser.add_argument('-printevery', type=int, default=5000)
    parser.add_argument('-window', type=int, default=3)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-dropout', type=int, default=0.35)
    parser.add_argument('-clip', type=int, default=2.0)
    parser.add_argument('-model', type=str,default='LSTM')
    parser.add_argument('-savename', type=str,default='lstm')
    parser.add_argument('-loadname', type=str)
    parser.add_argument('-trainname', type=str,default='wiki.train.txt')
    parser.add_argument('-validname', type=str,default='wiki.valid.txt')
    parser.add_argument('-testname', type=str,default='wiki.test.txt')

    params = {
        'd_model': 512,
        'd_hidden': 512,
        'n_layers': 2,
        'batch_size': 20,
        'seq_len': 30,
        'printevery': 500,
        'window': 3,
        'epochs': 20,
        'lr': 0.0001,
        'dropout': 0.35,
        'clip': 2.0,
        'model': 'FFNN',
        'savename': 'ffnn',
        'loadname': None,
        'trainname': './hw#1/mix.train.tok',
        'validname': './hw#1/mix.valid.tok',
        'testname': './hw#1/mix.test.tok'
    }
    parser.add_argument("-f", required=False)
    
    [vocab,words,train] = read_encode(params['trainname'],[],{},[],3)
    # print('vocab: %d train: %d' % (len(vocab),len(train)))
    [vocab,words,test] = read_encode(params['testname'],vocab,words,[],-1)
    # print('vocab: %d test: %d' % (len(vocab),len(test)))
    params['vocab_size'] = len(vocab)

    train_loader = read_encode(params['trainname'],[],{},[],3)
    model_lstm = LSTM(vocab,words,512,512,2,.35,True).to(device)
    model_ffnn = FFNN(vocab, words, d_model=100, d_hidden=100, dropout=0.2)

    if params['model'] == 'FFNN':
        [vocab,words,train] = read_encode_fnn(params['trainname'],[],{},[],3)
        print('vocab: %d train: %d' % (len(vocab), len(train)))
        [vocab,words,test] = read_encode_fnn(params['testname'], vocab,words,[],-1)
        print('vocab: %d test: %d' % (len(vocab),len(test)))
        params['vocab_size'] = len(vocab)
        loss_function = nn.NLLLoss()
        BATCH_SIZE = 64
        optimizer = torch.optim.Adam(model_ffnn.parameters(), lr=0.0003)
        train_perplexity_scores = []
        valid_perplexity_scores = []

        for epoch in range(50):
            print(epoch)
            train_running_loss_loss_func = 0
            train_running_loss_perplexity = 0
            valid_running_loss_loss_func = 0
            valid_running_loss_perplexity = 0

            train_split_bios = read_bios(params['trainname'], True)
            train_windows = create_windows(train_split_bios, 3)
            train_context_for_fake_bios, train_next_words_for_fake_bios = create_context_and_next_words(train_windows, words)
            train_dataset = TensorDataset(train_context_for_fake_bios, train_next_words_for_fake_bios)
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            for i, (context, label) in enumerate(train_dataloader):
                print(context)
                log_probabilities = model_ffnn(context)
                # Collapsing labels to correct dimensions
                label = label.squeeze()
                loss = loss_function(log_probabilities, label)

                true_word_probabilities = []
                for j in range(len(log_probabilities)):
                    current_true_prob = log_probabilities[j][label[j]]
                    true_word_probabilities.append(current_true_prob.item())

                train_running_loss_loss_func += loss.item()
                train_running_loss_perplexity += sum(true_word_probabilities) / BATCH_SIZE
                
                model_ffnn.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i == 499:
                    break

            valid_split_bios = read_bios(params['validname'], True)
            valid_windows = create_windows(valid_split_bios, 3)
            valid_context_for_fake_bios, valid_next_words_for_fake_bios = create_context_and_next_words(valid_windows, words)

            valid_dataset = TensorDataset(valid_context_for_fake_bios, valid_next_words_for_fake_bios)
            valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

            for _, (context, label) in enumerate(valid_dataloader):
                log_probabilities = model_ffnn(context)
                # Collapsing labels to correct dimensions
                label = label.squeeze()
                loss = loss_function(log_probabilities, label)

                true_word_probabilities = []
                for j in range(len(log_probabilities)):
                    current_true_prob = log_probabilities[j][label[j]]
                    true_word_probabilities.append(current_true_prob.item())

                valid_running_loss_loss_func += loss.item()
                valid_running_loss_perplexity += sum(true_word_probabilities) / BATCH_SIZE

            train_perplexity = math.exp(- train_running_loss_perplexity / 500)
            train_perplexity_scores.append(train_perplexity)

            valid_perplexity = math.exp(- valid_running_loss_perplexity / len(valid_dataloader))
            valid_perplexity_scores.append(valid_perplexity)
            
            print('epoch {}, training loss: {}, valid loss: {}'.format(epoch, train_running_loss_loss_func / 500, valid_running_loss_loss_func / len(valid_dataloader)))
            print('train perplexity: {}, valid perplexity: {}'.format(train_perplexity, valid_perplexity))      
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_ffnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_perplexity': train_perplexity_scores,
                'valid_perplexity': valid_perplexity_scores,
                }, "./model/train_ffnn.pt")
            
            # Stop
            if epoch >= 1 and valid_perplexity_scores[-1] >= valid_perplexity_scores[-2]:
                print("Overfitted")
                break
        print("Finished!")

        pyplot.plot(range(len(train_perplexity_scores)), train_perplexity_scores, label='Training')
        pyplot.plot(range(len(valid_perplexity_scores)), valid_perplexity_scores, label='Valid')

        pyplot.xlabel('Epochs')
        pyplot.ylabel('Perplexity Score')
        pyplot.title('Perplexity over Epochs')
        pyplot.legend()
        pyplot.show()
        
    if params['model'] == 'LSTM':
        encodings = read_encode('./hw#1/mix.train.tok',[],{},[],3) 
        train_data,vocab,words = read_alltext('./hw#1/mix.train.tok',20,encodings[0],encodings[1])
        valid_data,vocab,words = read_alltext('./hw#1/mix.valid.tok',20,vocab,words)
        test_data,vocab,words = read_alltext('./hw#1/mix.test.tok',20,vocab,words)

        criterion = nn.CrossEntropyLoss() 
        batch_size = 20
        seq_len = 30
        clip = 2.0
        n_epochs = 1
        saved = True
        
        optimizer = optim.Adam(model_lstm.parameters(), lr=.0008)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)

        train = []
        valid = []
        best_valid_loss = float('inf')

        for epoch in range(n_epochs):
            train_loss = LSTM_train(model_lstm, train_data, optimizer, criterion, 
                        20, seq_len, clip, device)
            valid_loss = evaluate(model_lstm, valid_data, criterion, 20, 
                        seq_len, device)
            
            lr_scheduler.step(valid_loss)

            print(f'\tTrain Perplexity: {train_loss:.3f}')
            print(f'\tValid Perplexity: {valid_loss:.3f}')

            train.append(train_loss)
            valid.append(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model_lstm.state_dict(), 'best-val-lstm_lm.pt')

        epochs = range(1, 26)
        
        plt.plot(epochs, train , label='Training Loss')
        plt.plot(epochs, valid, label='Validation Loss')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        
        plt.xticks(arange(0, 26, 2))
        plt.legend(loc='best')
        plt.show()

    if params['model'] == 'FFNN_CLASSIFY':
        [vocab,words,train] = read_encode_fnn(params['trainname'],[],{},[],3)
        print('vocab: %d train: %d' % (len(vocab), len(train)))
        [vocab,words,test] = read_encode_fnn(params['testname'], vocab, words,[],-1)
        print('vocab: %d test: %d' % (len(vocab),len(test)))
        split_bios = read_bios('./hw#1/mix.train.txt', True)
        windows = create_windows(split_bios, 3)
        
        all_sliding_windows = [item[0] for item in windows]
        all_bio_labels = [item[1] for item in windows]
        model_ffnn_final = FFNN(vocab, words, d_model=100, d_hidden=100, dropout=0.2)
        BATCH_SIZE = 64
        # print(torch.load('./model/final_ffnn.pt'))
        model_ffnn_final.load_state_dict(torch.load('./model/train_ffnn.pt')['model_state_dict'])
        model_ffnn_final.eval()

        print(model_ffnn_final)
        probabilities = []

        for i in range(len(all_sliding_windows)):
            all_sliding_windows_for_bio = all_sliding_windows[i]
            context, true_words = make_context_and_true_words_per_bio(all_sliding_windows_for_bio, words)

        bio_label = all_bio_labels[i]

        # Each bio has a probability table (mapping words to their probabilities, given the sequence)
        log_probability_tables = model_ffnn_final(context)

        normal_probabilities = torch.FloatTensor([])

        for idx_curr_context, probability_distributions_for_each_context in enumerate(log_probability_tables):
            true_word_for_curr_context = true_words[idx_curr_context]
            probability_for_true_word = probability_distributions_for_each_context[true_word_for_curr_context]
            probability_for_predicted_word = torch.max(probability_distributions_for_each_context)

            probability_normalized = (probability_for_predicted_word - probability_for_true_word) / (probability_for_predicted_word + 1e-9)

            normal_probabilities = torch.cat([normal_probabilities, probability_normalized])

        # Trimming mean by 0.05 (from each side) and dividing by the length
        trimmed_mean = stats.trim_mean(normal_probabilities.detach().numpy(), 0.05) / len(log_probability_tables)

        probabilities.append([trimmed_mean, bio_label])

        ## Evaluating on Training Data
        plt.style.use('seaborn-deep')

        probabilites_true = [item[0] for item in probabilities if item[1] == 1]
        probabilites_fake = [item[0] for item in probabilities if item[1] == 0]

        # min_bound = math.floor(min(min(probabilites_true), min(probabilites_fake)))

        x = probabilites_true
        y = probabilites_fake
        bins = np.linspace(-0.025, 0, 100)
        print(x)
        print(y)
        plt.hist([x, y], bins, label=['true', 'fake'])
        plt.legend(loc='upper left')
        plt.show()

        X, y = [item[0] for item in probabilities], [item[1] for item in probabilities]

        # split into train/test sets
        trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)
        trainX = np.array(trainX).reshape(-1, 1)
        testX = np.array(testX).reshape(-1, 1)

        # fit a model
        lg_model = LogisticRegression(solver='lbfgs')
        lg_model.fit(trainX, trainy)

        # predict probabilities
        yhat = lg_model.predict_proba(testX)
        # keep probabilities for the positive outcome only
        yhat = yhat[:, 1]

        # calculate roc curves
        fpr, tpr, thresholds = roc_curve(testy, yhat)

        # calculate the g-mean for each threshold
        gmeans = sqrt(tpr * (1-fpr))
        # locate the index of the largest g-mean
        idx_best = argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[idx_best], gmeans[idx_best]))
        best_threshold = thresholds[idx_best]
        # plot the roc curve for the model
        pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
        pyplot.plot(fpr, tpr, marker='.', label='Logistic')
        pyplot.scatter(fpr[idx_best], tpr[idx_best], marker='o', color='black', label='Best')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.legend()
        # show the plot
        pyplot.show()

        # predict probabilities
        yhat = lg_model.predict_proba(testX)
        # keep probabilities for the positive outcome only
        yhat = yhat[:, 1]

        y_predictions = yhat >= thresholds[idx_best]
        y_predictions = [y.astype(int) for y in y_predictions]

        print(confusion_matrix(testy, y_predictions))
        print(accuracy_score(testy, y_predictions))

    if params['model'] == 'LSTM_CLASSIFY':
        encodings = read_encode('./hw#1/mix.train.tok',[],{},[],3) 
        train_data,vocab,words = read_alltext('./hw#1/mix.train.tok',20,encodings[0],encodings[1])
        valid_data,vocab,words = read_alltext('./hw#1/mix.valid.tok',20,vocab,words)
        test_data,vocab,words = read_alltext('./hw#1/mix.test.tok',20,vocab,words)
        model_lstm.load_state_dict(torch.load('./model/lstm_mixed.pt',  map_location=device)['model_state_dict'])
        criterion = nn.CrossEntropyLoss() 
        batch_size = 20
        seq_len = 30
        clip = 2.0
        n_epochs = 25
        test_loss = evaluate(model_lstm, test_data, criterion, batch_size, seq_len, device)
        print(f'Test Perplexity: {test_loss:.3f}')
        print(params)

        word_tensors = make_tensors('./hw#1/mix.test.tok',vocab,words)
        input = torch.tensor(word_tensors[0][100],device = device).reshape(-1,1)

        y_true=[] 
        y_pred=[]
        correct = 0
        for index in range(len(word_tensors[1])):
            item = torch.tensor(word_tensors[0][index],device = device).reshape(-1,1)
            prediction = LSTM_Classify(item,model_lstm,vocab,words,device,seq_len)
            actual = word_tensors[1][index] 
            if(prediction == 637):
                y_pred.append(1)
            else:
                y_pred.append(0)
            if(actual == 637):
                y_true.append(1)
            else:
                y_true.append(0)
            if prediction == actual:
                correct += 1
            
            # print(correct)
            print(correct/len(word_tensors[1]))

        confusion_matrix(y_true, y_pred)

if __name__ == "__main__":
    main()