#Base on PGM Homework

import torch
import torch.nn as nn
from sklearn import metrics


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, state_size, embedding_dim, feature_dim):
        """Linear-chain CRF with Bi-directional LSTM features.

        Args:
            vocab_size (int): num of word types in dictionary
            state_size (int): num of states
            embedding_dim (int): word embedding dim size
            feature_dim (int): word feature dim size
        """
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.state_size = state_size
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim

        # word features
        # obs_seq [seq, batch] -> embed_seq [seq, batch, embed]
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        # embed_seq [seq, batch, embed] -> bi-LSTM feature_seq [seq, batch, dir * feat]
        self.lstm = nn.LSTM(embedding_dim, self.feature_dim, num_layers=1, bidirectional=True)

        # singleton potential
        # feature_seq [seq, batch, dir * feat] -> singleton potential [seq, batch, state]
        self.linear_transform = nn.Linear(2 * self.feature_dim, self.state_size)

        # pairwise potential
        # [state at time t-1, state at time t], independent of obs
        self.pairwise_score = nn.Parameter(torch.randn(self.state_size, self.state_size))
        self.start_score = nn.Parameter(torch.randn(self.state_size))
        self.stop_score = nn.Parameter(torch.randn(self.state_size))

    def lstm_features(self, obs_seq):
        """Generate features for each observation in the sequence using LSTM.

        Args:
            obs_seq (torch.LongTensor): [seq, batch]

        Returns:
            feature_seq (torch.FloatTensor): [seq, batch, dir * feat]
        """
        seq, batch = obs_seq.shape
        embed_seq = self.embedding(obs_seq)  # [seq, batch, embed]
        h_in = torch.randn(2, batch, self.feature_dim)  # [layer * dir, batch, feat]
        c_in = torch.randn(2, batch, self.feature_dim)  # [layer * dir, batch, feat]
        feature_seq, _ = self.lstm(embed_seq, (h_in, c_in))  # [seq, batch, dir * feat]
        return feature_seq

    def log_partition(self, feature_seq):
        """Compute log partition function log Z(x_{1:T}) for features phi_{1:T}.

        Args:
            feature_seq (torch.FloatTensor): [seq, batch, dir * feat]
        Returns:
            log_partition (torch.FloatTensor): [batch]
        """

        single_score = self.linear_transform(feature_seq) #seq batch state
        alph = single_score[0] + self.start_score #batch state
        # p(zt-1)* p(zt|zt-1) * p(zt|xt)
        for i in range(1, single_score.shape[0]):
            alph =  single_score[i] + torch.logsumexp( torch.add(alph.unsqueeze(2), self.pairwise_score), dim = 1) # batch * state 
        alph = torch.add(alph, self.stop_score) 
        return torch.logsumexp(alph, dim = 1) #batch


    def log_score(self, feature_seq, state_seq):
        """Compute log score p_tilde(z_{1:T}, x_{1:T}) for features phi_{1:T} and states z_{1:T}.

        Args:
            feature_seq (torch.FloatTensor): [seq, batch, dir * feat]
            state_seq (torch.LongTensor): [seq, batch]

        Returns:
            log_score (torch.FloatTensor): [batch]
        """
        seq, batch, _ = feature_seq.shape
        singleton_score = self.linear_transform(feature_seq)  # [seq, batch, state]
        score = singleton_score[0, torch.arange(batch), state_seq[0]]  # [batch]
        score += self.start_score[state_seq[0]]  # [batch]
        for t in range(1, seq):
            score += singleton_score[t, torch.arange(batch), state_seq[t]]  # [batch]
            score += self.pairwise_score[state_seq[t-1], state_seq[t]]  # [batch]
        score += self.stop_score[state_seq[-1]]  # [batch]
        return score

    def neg_log_likelihood(self, obs_seq, state_seq):
        """Compute negative log-likelihood for observation x_{1:T} and states z_{1:T}.

        Args:
            obs_seq (torch.LongTensor): [seq, batch]
            state_seq (torch.LongTensor): [seq, batch]

        Returns:
            nll (float): negative log likelihood of the input
        """
        feature_seq = self.lstm_features(obs_seq)  # [seq, batch, dir * feat]
        log_partition = self.log_partition(feature_seq)  # [batch]
        log_score = self.log_score(feature_seq, state_seq)  # [batch]
        nll = torch.mean(log_partition - log_score)  # float
        return nll

    def viterbi_decode(self, feature_seq):
        """Viterbi decoding for features phi_{1:T}.

        Args:
            feature_seq (torch.FloatTensor): [seq, batch, dir * feat]

        Returns:
            max_score (torch.FloatTensor): [batch]
            state_seq (torch.LongTensor): [seq, batch]
        """
        # TODO: Implement this
        single_score = self.linear_transform(feature_seq) # seq, batch, state
        state_seq = torch.zeros(single_score.shape[0], single_score.shape[1], dtype= torch.int64)
        #maxscore matrix
        alph = torch.randn(single_score.shape[0], single_score.shape[1], single_score.shape[2]) # seq, batch, state 
        #best path matrix  first row is ignored
        alph_plus = torch.zeros(single_score.shape[0], single_score.shape[1], single_score.shape[2], dtype= torch.int64) # seq, batch, state 
        lastone = (single_score[0]) + (self.start_score) # batch, state
        alph[0] = lastone # batch * state
        for i in range(1, single_score.shape[0]):
            score = torch.add(lastone.unsqueeze(2), self.pairwise_score)
            value, value_index = torch.max(score, dim = 1)
            lastone  = value + single_score[i]
            alph_plus[i] = value_index
            alph[i] = lastone
        lastone += self.stop_score
        max_score,_ = torch.max(lastone, dim = 1) #batch
        state_seq[-1] = torch.argmax(lastone, dim = 1) # why torch.argmax return a float
        #backdecoding
        for i in range(state_seq.shape[0] - 1, 0, -1):
            for j in range(state_seq.shape[1]):
                state_seq[i - 1, j] = alph_plus[i, j, state_seq[i,j]]
        return max_score, state_seq

    def forward(self, obs_seq):
        """

        Args:
            obs_seq (torch.LongTensor): [seq, batch]

        Returns:
            max_score (torch.FloatTensor): [batch]
            state_seq (torch.LongTensor): [seq, batch]
        """
        feature_seq = self.lstm_features(obs_seq)  # [seq, batch, dir * feat]
        max_score, state_seq = self.viterbi_decode(feature_seq)  # [batch], [seq, batch]
        return max_score, state_seq

