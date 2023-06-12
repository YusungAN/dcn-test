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
    vec_tfidf = TfidfVectorizer(max_features=10*10)
    tfidf_train = vec_tfidf.fit_transform(train_data['content'].tolist()).todense()

    return tfidf_train


def bert_embedding(text, model):
    text = str(text)
    print(text)
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    if len(indexed_tokens) > 512:
        indexed_tokens = indexed_tokens[:512]
        segments_ids = segments_ids[:512]
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        tokens_tensor = tokens_tensor.to('cuda')
        segments_tensors = segments_tensors.to('cuda')
        print(len(tokens_tensor[0]))
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding


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
    ns_yoga = pd.read_csv('dcn-test/ns_yoga_preprocessed.csv')
    ns_yoga['label'] = labels
    ns_yoga.to_csv('ns_yoga_labeled.csv')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Deep Clustering Network')

    # Dataset parameters
    parser.add_argument('--dir', default='../Dataset/mnist',
                        help='dataset directory')
    parser.add_argument('--input-dim', type=int, default=768,
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
    # tfidf = torch.tensor(get_tfidf_data(ns_yoga)).type(torch.float32)
    # dataset = torch.utils.data.TensorDataset(tfidf)
    # train_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=args.batch_size, shuffle=False
    # )

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
    bert_model.cuda()
    bert_model.eval()
    ns_yoga.dropna(subset=['content'], inplace=True)
    td = [0] * len(ns_yoga.index)
    td = ns_yoga['content'].apply(bert_embedding)
    bert_embedding = torch.tensor(td).type(torch.float32)
    dataset = torch.utils.data.TensorDataset(td)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    # Main body
    model = DCN(args)
    solver(args, model, train_loader)