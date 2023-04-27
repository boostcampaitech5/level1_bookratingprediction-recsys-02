import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

def dl_data_load(args):
    """
    Parameters
    ----------
    Args:
        data_path : str
            데이터 경로
    ----------
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.data_path + 'users_final.csv')
    books = pd.read_csv(args.data_path + 'books_final.csv')
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    # test = pd.read_csv(args.data_path + 'test_ratings_filtered.csv')
    # sub = pd.read_csv(args.data_path + 'sample_submission_filtered.csv')
    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.data_path + 'sample_submission.csv')

    users['user_id'] = users['user_id'].astype(str)
    users['binning_age'] = users['binning_age'].astype('str')
    train['user_id'] = train['user_id'].astype(str)
    test['user_id'] = test['user_id'].astype(str)
    sub['user_id'] = sub['user_id'].astype(str)

    users['binning_age'] = users['binning_age'].fillna(value = 'other')
    users['location_city'] = users['location_city'].fillna(value = 'other')
    users['location_state'] = users['location_state'].fillna(value = 'other')
    users['location_country'] = users['location_country'].fillna(value = 'other')
    books['pnumber'] = books['pnumber'].fillna(value = 'other')
    books['book_title'] = books['book_author'].fillna(value = 'other')
    books['book_author'] = books['book_author'].fillna(value = 'other')
    books['language'] = books['language'].fillna(value = 'other')
    books['binning_year'] = books['binning_year'].fillna(value = 'other')
    books['category_high'] = books['category_high'].fillna(value = 'other')

    train = train.merge(books, how='left', on='isbn')
    train = train.merge(users, how='left', on='user_id')
    test = test.merge(books, how='left', on='isbn')
    test = test.merge(users, how='left', on='user_id')
    sub = sub.merge(books, how='left', on='isbn')
    sub = sub.merge(users, how='left', on='user_id')

    features = ['user_id', 'isbn', 'book_author', 'language', 'category_high','pnumber', \
                'binning_year','binning_age', 'location_city', 'location_state', 'location_country', 'book_title']
    train = train[features + ['rating']]
    test = test[features + ['rating']]
    sub = test[features + ['rating']]

    # ids = train['user_id'].unique()
    # isbns = train['isbn'].unique()
    # authors = train['book_author'].unique()
    # languages = train['language'].unique()
    # categories = train['category_high'].unique()
    # pnumbers = train['pnumber'].unique()
    # years = train['binning_year'].unique()
    # ages = train['binning_age'].unique()
    # cities = train['location_city'].unique()
    # states = train['location_state'].unique()
    # countries = train['location_country'].unique()
    # titles = train['book_title'].unique()

    ids = users['user_id'].unique()
    isbns = books['isbn'].unique()
    authors = books['book_author'].unique()
    languages = books['language'].unique()
    categories = books['category_high'].unique()
    pnumbers = books['pnumber'].unique()
    years = books['binning_year'].unique()
    ages = users['binning_age'].unique()
    cities = users['location_city'].unique()
    states = users['location_state'].unique()
    countries = users['location_country'].unique()
    titles = books['book_title'].unique()
    
    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}
    idx2author = {idx:id for idx, id in enumerate(authors)}
    idx2category = {idx:isbn for idx, isbn in enumerate(categories)}
    idx2language = {idx:isbn for idx, isbn in enumerate(languages)}
    idx2pnumber = {idx:id for idx, id in enumerate(pnumbers)}
    idx2year = {idx:isbn for idx, isbn in enumerate(years)}
    idx2age = {idx:id for idx, id in enumerate(ages)}
    idx2city = {idx:isbn for idx, isbn in enumerate(cities)}
    idx2state = {idx:id for idx, id in enumerate(states)}
    idx2country = {idx:isbn for idx, isbn in enumerate(countries)}
    idx2title = {idx:id for idx, id in enumerate(titles)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}
    author2idx = {id:idx for idx, id in idx2author.items()}
    category2idx = {isbn:idx for idx, isbn in idx2category.items()}
    language2idx = {isbn:idx for idx, isbn in idx2language.items()}
    pnumber2idx = {id:idx for idx, id in idx2pnumber.items()}
    year2idx = {isbn:idx for idx, isbn in idx2year.items()}
    age2idx = {id:idx for idx, id in idx2age.items()}
    city2idx = {isbn:idx for idx, isbn in idx2city.items()}
    state2idx = {id:idx for idx, id in idx2state.items()}
    country2idx = {isbn:idx for idx, isbn in idx2country.items()}
    title2idx = {id:idx for idx, id in idx2title.items()}

    feature2mapDict = {'user_id' : user2idx, 'isbn' : isbn2idx, 'book_author' : author2idx, 'language' : language2idx, \
    'category_high' : category2idx, 'pnumber' : pnumber2idx, 'binning_year' : year2idx,'binning_age' : age2idx, \
    'location_city' : city2idx, 'location_state' : state2idx, 'location_country' : country2idx, 'book_title' : title2idx}

    for feature, mapDict in feature2mapDict.items():
        train[feature] = train[feature].map(mapDict)
        test[feature] = test[feature].map(mapDict)
        sub[feature] = sub[feature].map(mapDict)

    field_dims = np.array([len(user2idx), len(isbn2idx), len(author2idx), len(language2idx), len(category2idx), len(pnumber2idx),\
                len(year2idx), len(age2idx), len(city2idx), len(state2idx), len(country2idx), len(title2idx)], dtype=np.uint32)

    data = {
            'train':train,
            'test':test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            'idx2author':idx2author,
            'idx2language':idx2language,
            'author2idx':author2idx,
            'language2idx':language2idx,
            'idx2category':idx2category,
            'idx2pnumber':idx2pnumber,
            'category2idx':category2idx,
            'pnumber2idx':pnumber2idx,
            'idx2year':idx2year,
            'idx2age':idx2age,
            'year2idx':year2idx,
            'age2idx':age2idx,
            'idx2city':idx2city,
            'idx2state':idx2state,
            'city2idx':city2idx,
            'state2idx':state2idx,
            'idx2country':idx2country,
            'idx2title':idx2title,
            'country2idx':country2idx,
            'title2idx':title2idx,
            }

    return data

def dl_data_split(args, data):
    """
    Parameters
    ----------
    Args:
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            랜덤 seed 값
    ----------
    """

    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=True,
                                                        stratify = data['train']['rating']
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def dl_data_loader(args, data):
    """
    Parameters
    ----------
    Args:
        batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        data_shuffle : bool
            data shuffle 여부
    ----------
    """

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
