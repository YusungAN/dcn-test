import torch
import argparse
import numpy as np
from DCN import DCN
from torchvision import datasets, transforms
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd

def evaluate(model, test_loader, label_list):
    lat_x = 0
    for data in test_loader:
        data = data[0].to(model.device)
        batch_size = data.size()[0]
        data = data.view(batch_size, -1)
        latent_X = model.autoencoder(data, latent=True)
        latent_X = latent_X.detach().cpu().numpy()

        label_x = model.kmeans.update_assign(latent_X)
        label_list = np.hstack((label_list, label_x))

    return lat_x


def get_tfidf_data(train_data):

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec_tfidf = TfidfVectorizer(max_features=10*10)
    tfidf_train = vec_tfidf.fit_transform(train_data['content'].tolist()).todense()

    return tfidf_train


def solver(args, model, train_loader):

    rec_loss_list = model.pretrain(train_loader, args.pre_epoch)
    label_li = []
    for e in range(args.epoch):
        model.train()
        model.fit(e, train_loader)

        model.eval()

        lat_x = evaluate(model, train_loader, label_li)  # evaluation on test_loader

        print('\nEpoch: {:02d} | example label: {}\n'.format(e, lat_x))
        if e < args.epoch-1:
            label_li = []
    labels = pd.Series(label_li)
    ns_yoga = pd.read_csv('dcn-test/ns_yoga_preprocessed.csv')
    ns_yoga['label'] = labels
    ns_yoga.to_csv('ns_yoga_labeled.csv')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Deep Clustering Network')

    # Dataset parameters
    parser.add_argument('--dir', default='../Dataset/mnist',
                        help='dataset directory')
    parser.add_argument('--input-dim', type=int, default=10*10,
                        help='input dimension')
    parser.add_argument('--n-classes', type=int, default=4,
                        help='output dimension')

    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--epoch', type=int, default=30,
                        help='number of epochs to train')
    parser.add_argument('--pre-epoch', type=int, default=50, 
                        help='number of pre-train epochs')
    parser.add_argument('--pretrain', type=bool, default=True,
                        help='whether use pre-training')

    # Model parameters
    parser.add_argument('--lamda', type=float, default=1,
                        help='coefficient of the reconstruction loss')
    parser.add_argument('--beta', type=float, default=1,
                        help=('coefficient of the regularization term on '
                              'clustering'))
    parser.add_argument('--hidden-dims', default=[500, 500, 2000],
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='latent space dimension')
    parser.add_argument('--n-clusters', type=int, default=4,
                        help='number of clusters in the latent space')

    # Utility parameters
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='number of jobs to run in parallel')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='whether to use GPU')
    parser.add_argument('--log-interval', type=int, default=100,
                        help=('how many batches to wait before logging the '
                              'training status'))

    args = parser.parse_args()

    ns_yoga = pd.read_csv('dcn-test/ns_yoga_preprocessed.csv')
    ns_yoga.dropna(subset=['content'], inplace=True)
    tfidf = torch.tensor(get_tfidf_data(ns_yoga)).type(torch.float32)
    dataset = torch.utils.data.TensorDataset(tfidf)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    # Main body
    model = DCN(args)
    solver(args, model, train_loader)