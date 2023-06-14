import torch
import argparse
import numpy as np
from DCN import DCN
from torchvision import datasets, transforms
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import pickle
import gdown

def evaluate(model, test_loader):
    label_list = []
    for data in test_loader:
        data = data[0].to(model.device)
        batch_size = data.size()[0]
        data = data.view(batch_size, -1)
        latent_X = model.autoencoder(data, latent=True)
        latent_X = latent_X.detach().cpu().numpy()

        label_x = model.kmeans.update_assign(latent_X)
        label_list = np.hstack((label_list, label_x))

    return label_list


def get_tfidf_data(train_data):

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec_tfidf = TfidfVectorizer(max_features=768)
    tfidf_train = vec_tfidf.fit_transform(train_data.tolist()).todense()

    return tfidf_train


def solver(args, model, train_loader):
    label_li = []
    rec_loss_list = model.pretrain(train_loader, args.pre_epoch)
    for e in range(args.epoch):
        model.train()
        model.fit(e, train_loader)
        model.eval()

        label_li = evaluate(model, train_loader)  # evaluation on test_loader
        print(label_li)
    labels = pd.Series(label_li)
    review_df = pd.read_csv('dcn-test/ns_review_txt1_drop_dup.csv', lineterminator='\n')
    for i in range(2, 9):
        tmp_df = pd.read_csv('dcn-test/ns_review_txt{}_drop_dup.csv'.format(i), lineterminator='\n')
        review_df = pd.concat([review_df, tmp_df])

    review_df['label'] = labels
    review_df.to_csv('full_reviews_bert_labeled.csv')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Deep Clustering Network')

    # Dataset parameters
    parser.add_argument('--dir', default='../Dataset/mnist',
                        help='dataset directory')
    parser.add_argument('--input-dim', type=int, default=768,
                        help='input dimension')
    parser.add_argument('--n-classes', type=int, default=10,
                        help='output dimension')

    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--epoch', type=int, default=50,
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
    parser.add_argument('--n-clusters', type=int, default=10,
                        help='number of clusters in the latent space')

    # Utility parameters
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='number of jobs to run in parallel')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='whether to use GPU')
    parser.add_argument('--log-interval', type=int, default=100,
                        help=('how many batches to wait before logging the '
                              'training status'))
    parser.add_argument('--pickle_download', type=int, default=0,
                        help='if 0, do not download embedding vector file')
    parser.add_argument('--use_tfidf', type=int, default=0,
                        help('if 1, use tfidf, not bert'))

    args = parser.parse_args()


    if args.pickle_download:
        google_path = 'https://drive.google.com/uc?id='
        file_id = ['1_Zh-yrJM5e1M00R_A9L_pgEJuT-__oZZ',
                   '1IucMhCRbo7XFBYGi5cN3L23iNwkoOmzE',
                   '1eotABBI3LLzmtn87FJe8gkakleSQs-5q',
                   '1oHilp0eTurx9qLN8FIoRO6ZDq_azEJgO',
                   '1HxRMyx-WaPm-N8n-76X5ONRKXawvVX8f',
                   '1BBhyshJQT3dZ4W_UCSgk4S4FmM5FyjIf',
                   '1chhsiJuMGZk0daTgu0K1uLUlH-_VYksz',
                   '1bwvlRciJgOVFBqwkhGWCXOZm7O5fQfIB']
        for i in range(1, 9):
            output_name = "bert_embedding_tensor{}.pickle".format(i)
            gdown.download(google_path+file_id[i-1], output_name, quiet=False)
            print(i)

    train_loader = 0
    if args.use_tfidf == 1:
        review_df = pd.read_csv('dcn-test/ns_review_txt1_drop_dup.csv', lineterminator='\n')
        for i in range(2, 9):
            tmp_df = pd.read_csv('dcn-test/ns_review_txt{}_drop_dup.csv'.format(i), lineterminator='\n')
            review_df = pd.concat([review_df, tmp_df])
        embedded_data = torch.tensor(get_tfidf_data(review_df['content'])).type(torch.float32)
    else:
        with open("bert_embedding_tensor1.pickle", "rb") as fr:
            data = pickle.load(fr)

        full_bert_embedding = data
        print(1)
        for i in range(2, 9):
            with open("bert_embedding_tensor{}.pickle".format(i), "rb") as fr:
                data = pickle.load(fr)
            print(i)
            full_bert_embedding = torch.cat([full_bert_embedding, data], dim=0)

        # embedded_data = torch.tensor(embedded_data).type(torch.float32)
        dataset = torch.utils.data.TensorDataset(full_bert_embedding)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False
        )

    # Main body
    model = DCN(args)
    solver(args, model, train_loader)