import argparse
import numpy as np
from DCN import DCN
import pandas as pd
import torch
import pickle
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertConfig, BertModel

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
    vec_tfidf = TfidfVectorizer(max_features=100)
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
        print(set(label_li))
    labels = pd.Series(label_li)
    '''
    review_df = pd.read_csv('dcn-test/ns_review_txt1_drop_dup.csv', lineterminator='\n').iloc[:10000]
    for i in range(2, 9):
        tmp_df = pd.read_csv('dcn-test/ns_review_txt{}_drop_dup.csv'.format(i), lineterminator='\n').iloc[:10000]
        review_df = pd.concat([review_df, tmp_df])
    '''
    with open("review_min_df", "rb") as fr:
        review_df = pickle.load(fr)
    review_df['label'] = labels
    review_df.to_csv('reviews_ksbert_labeled.csv')

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
    parser.add_argument('--lr', type=float, default=2e-4,
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
    parser.add_argument('--hidden-dims', default=[192, 192, 768],
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
                        help='if 1, use tfidf, not bert')

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
        review_df = pd.read_csv('dcn-test/ns_review_txt1_drop_dup.csv', lineterminator='\n')['content']
        embedded_data = torch.tensor(get_tfidf_data(review_df))
        for i in range(2, 9):
            tmp_df = pd.read_csv('dcn-test/ns_review_txt{}_drop_dup.csv'.format(i), lineterminator='\n')
            tmp_data = torch.tensor(get_tfidf_data(tmp_df['content'])).type(torch.float32)
            embedded_data = torch.cat([embedded_data, tmp_data], dim=0)
            print(i)

        dataset = torch.utils.data.TensorDataset(embedded_data)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False
        )
        print('end')
    else:
        tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base", do_lower_case=False)
        pretrained_model_config = BertConfig.from_pretrained("beomi/kcbert-base")
        model = BertModel.from_pretrained("beomi/kcbert-base", config=pretrained_model_config)

        with open("data.pickle", "rb") as fr:
            sentences = pickle.load(fr)

        features = tokenizer(
            sentences,
            max_length=512,
            padding="max_length",
            truncation=True,
        )

        features = {k: torch.tensor(v) for k, v in features.items()}

        outputs = model(**features)
        dataset = torch.utils.data.TensorDataset(outputs.last_hidden_state)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False
        )

    # Main body
    model = DCN(args)
    solver(args, model, train_loader)