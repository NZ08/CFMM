import torch
import torch.nn as nn

'''
CBN (Conditional Batch Normalization layer)
    uses an MLP to predict the beta and gamma parameters in the batch norm equation
    Reference : https://papers.nips.cc/paper/7237-modulating-early-visual-processing-by-language.pdf
'''
class CBN(nn.Module):

    def __init__(self, lstm_size, emb_size, out_size, use_betas=True, use_gammas=True, eps=1.0e-5):
        super(CBN, self).__init__()

        self.lstm_size = lstm_size # size of the lstm emb which is input to MLP
        self.emb_size = emb_size # size of hidden layer of MLP
        self.out_size = out_size # output of the MLP - for each channel
        self.use_betas = use_betas
        self.use_gammas = use_gammas

        self.batch_size = 64 #image_adj lstm_size;image_embeds 64
        self.channels = lstm_size


        # beta and gamma parameters for each channel - defined as trainable parameters
        self.betas = nn.Parameter(torch.zeros(self.channels, self.batch_size).cuda())
        self.gammas = nn.Parameter(torch.ones(self.channels, self.batch_size).cuda())
        self.eps = eps

        # MLP used to predict betas and gammas
        self.fc_gamma = nn.Sequential(
            # nn.Linear(self.lstm_size, self.emb_size),
            # nn.ReLU(inplace=True),
            # nn.Linear(self.emb_size, self.out_size),
            nn.Linear(self.lstm_size, self.out_size), #houjia
            ).cuda()

        self.fc_beta = nn.Sequential(
            # nn.Linear(self.lstm_size, self.emb_size),
            # nn.ReLU(inplace=True),
            # nn.Linear(self.emb_size, self.out_size),
            nn.Linear(self.lstm_size, self.out_size), #houjia
            ).cuda()

        # initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    '''
    Predicts the value of delta beta and delta gamma for each channel

    Arguments:
        lstm_emb : lstm embedding of the question

    Returns:
        delta_betas, delta_gammas : for each layer
    '''
    def create_cbn_input(self, lstm_emb):

        if self.use_betas:
            delta_betas = self.fc_beta(lstm_emb.T)
        else:
            delta_betas = torch.zeros(self.channels, self.batch_size).cuda()

        if self.use_gammas:
            delta_gammas = self.fc_gamma(lstm_emb.T)
        else:
            delta_gammas = torch.zeros(self.channels, self.batch_size).cuda()

        return delta_betas, delta_gammas

    '''
    Computer Normalized feature map with the updated beta and gamma values

    Arguments:
        feature : feature map from the previous layer
        lstm_emb : lstm embedding of the question

    Returns:
        out : beta and gamma normalized feature map
        lstm_emb : lstm embedding of the question (unchanged)

    Note : lstm_emb needs to be returned since CBN is defined within nn.Sequential
           and subsequent CBN layers will also require lstm question embeddings
    '''
    def forward(self, feature, lstm_emb):
        self.channels, self.batch_size = feature.data.shape

        # get delta values
        delta_betas, delta_gammas = self.create_cbn_input(lstm_emb)

        betas_cloned = self.betas.clone()
        gammas_cloned = self.gammas.clone()

        # update the values of beta and gamma
        betas_cloned += delta_betas.T
        gammas_cloned += delta_gammas.T

        # delta_feature_betas, delta_feature_gammas = self.create_cbn_input(feature) #houjia

        # betas_feature_cloned = self.betas.clone() #houjia
        # gammas_feature_cloned = self.gammas.clone()

        # update the values of beta and gamma
        # betas_feature_cloned += delta_feature_betas.T #houjia
        # gammas_feature_cloned += delta_feature_gammas.T

        batch_lstm_mean = torch.mean(lstm_emb)  # houjia
        batch_lstm_var = torch.var(lstm_emb)

        # get the mean and variance for the batch norm layer
        batch_mean = torch.mean(feature)  #yuan
        batch_var = torch.var(feature)

        # extend the betas and gammas of each channel across the height and width of feature map
        # betas_expanded = torch.stack([betas_cloned]*self.channels, dim=0)
        # betas_expanded = torch.stack([betas_expanded]*2, dim=1)
        #
        # gammas_expanded = torch.stack([gammas_cloned]*self.channels, dim=0)
        # gammas_expanded = torch.stack([gammas_expanded]*2, dim=1)

        # normalize the feature map
        feature_normalized = (feature-batch_mean)/torch.sqrt(batch_var+self.eps) #yuan
        # feature_normalized = feature
        lstm_normalized = (lstm_emb-batch_lstm_mean)/torch.sqrt(batch_lstm_var+self.eps)

        # get the normalized feature map with the updated beta and gamma values
        # out_feature = torch.mul(lstm_normalized, gammas_feature_cloned) + betas_feature_cloned#houjia
        out = torch.mul(feature_normalized, gammas_cloned) + betas_cloned
        out = 0.6*out + 0.2*feature_normalized + 0.2*lstm_normalized
        return out, lstm_emb

'''
# testing code
if __name__ == '__main__':
    torch.cuda.set_device(int(sys.argv[1]))
    model = CBN(512, 256)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            print 'found anomaly'
        if isinstance(m, nn.Linear):
            print 'found correct'
'''
