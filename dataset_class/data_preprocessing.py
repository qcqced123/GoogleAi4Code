import re, gc, glob, io, tokenize, markdown
import pandas as pd
import numpy as np
import torch
import configuration as configuration
from torch import Tensor
from bs4 import BeautifulSoup
from sklearn.model_selection import KFold, GroupKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from tqdm.auto import tqdm


def add_special_token(cfg: configuration.CFG, token: str) -> None:
    """
    Add special token to pretrained tokenizer
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        token: str, special token to add
    """
    special_token = token
    special_tokens_dict = {'additional_special_tokens': [f'{special_token}']}
    cfg.tokenizer.add_special_tokens(special_tokens_dict)
    tar_token_id = cfg.tokenizer(f'{special_token}', add_special_tokens=False)['input_ids'][0]

    setattr(cfg.tokenizer, 'tar_token', f'{special_token}')
    setattr(cfg.tokenizer, 'tar_token_id', tar_token_id)
    cfg.tokenizer.save_pretrained(f'{cfg.checkpoint_dir}/tokenizer/')


def tokenizing(cfg: configuration.CFG, text: str) -> any:
    """
    Preprocess text for CLIP
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        text: text from dataframe or any other dataset, please pass str type
    """
    inputs = cfg.tokenizer(
        text,
        max_length=cfg.max_len,
        padding='max_length',
        truncation=True,
        return_tensors=None,
        add_special_tokens=True,
    )
    for k, v in inputs.items():
        inputs[k] = torch.as_tensor(v)
    return inputs


def markdown_to_text(markdown_string: str) -> str:
    """
    Converts a markdown string to plaintext by beautifulsoup
    md -> html -> string
    Args:
        markdown_string: str, markdown string
    Example:
        markdown_to_text(md.loc['63a93277', 'source'])
        => md == pd.DataFrame filtered by cell_type == 'markdown'
    Reference:
        https://gist.github.com/lorey/eb15a7f3338f959a78cc3661fbc255fe
    """
    try:
        html = markdown.markdown(markdown_string)
        html = re.sub(r'<pre>(.*?)</pre>', ' ', html)  # remove code snippets
        html = re.sub(r'<code>(.*?)</code >', ' ', html)  # remove code snippets
        soup = BeautifulSoup(html, "html.parser")  # extract text
        text = ''.join(soup.findAll(text=True)).strip()  # extract text
        if len(text) == 0:
            text = markdown_string
            if text[0] == "!" and text[1] == "[":
                for m in range(2, len(text)):
                    if text[m] == "]":
                        text = 'embedded ' + text[2:m] + ' image'
                        break
            elif '<img src' in markdown_string or '.png' in markdown_string or 'gif' in markdown_string or '.jpg' in markdown_string:
                text = 'embedded image'
    except:
        text = markdown_string
    return text


def code_tokenizer(code: str) -> str:
    """
    Tokenize code text by python built-in tokenizer for code scanning
    Args:
        code: str, code text
    Example:
        code = code.loc['3a6623e3','source']
        code_text = tokenize.generate_tokens(io.StringIO(code).readline)
        ' '.join([tok.string for tok in code_text if tok.type==1 or tok.type==2 or tok.type==3 or tok.type==60])
    Reference:
        https://docs.python.org/3/library/tokenize.html
        https://www.kaggle.com/code/haithamaliryan/ai4code-extract-all-functions-variables-names/notebook
    """
    try:
        code_text = tokenize.generate_tokens(io.StringIO(code).readline)
        code_str = ' '.join([tok.string for tok in code_text if tok.type == 1 or tok.type == 2 or tok.type == 3 or tok.type == 60])
        if len(code_str) == 0:
            code_str = "unknown"
    except:
        code_str = code
    return code_str


def sequence_length(cfg: configuration.CFG, text_list: list) -> list:
    """ Get sequence length of all text data for checking statistics value """
    length_list = []
    for text in text_list:
        tmp_text = tokenizing(cfg, text)['attention_mask']
        length_list.append(tmp_text.count(1))
    return length_list


def check_null(df: pd.DataFrame) -> pd.Series:
    """ check if input dataframe has null type object...etc """
    return df.isnull().sum()


def kfold(df: pd.DataFrame, cfg: configuration.CFG) -> pd.DataFrame:
    """ KFold """
    fold = KFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed
    )
    df['fold'] = -1
    for num, (tx, vx) in enumerate(fold.split(df)):
        df.loc[vx, "fold"] = int(num)
    return df


def group_kfold(df: pd.DataFrame, cfg: configuration.CFG) -> pd.DataFrame:
    """ GroupKFold """
    fold = GroupKFold(
        n_splits=cfg.n_folds,
    )
    df['fold'] = -1
    for num, (tx, vx) in enumerate(fold.split(X=df, y=df['pct_rank'], groups=df['ancestor_id'])):
        df.loc[vx, "fold"] = int(num)
    return df


def mls_kfold(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """ Multilabel Stratified KFold """
    tmp_df = df.copy()
    y = pd.get_dummies(data=tmp_df.iloc[:, 2:8], columns=tmp_df.columns[2:8])
    fold = MultilabelStratifiedKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed
    )
    for num, (tx, vx) in enumerate(fold.split(X=df, y=y)):
        df.loc[vx, "fold"] = int(num)
    del tmp_df
    gc.collect()
    return df


def read_notebook(path) -> pd.DataFrame:
    """
    Make DataFrame which is subset of whole dataset from JSON file
    Options:
        pd.DataFrame.assign: make new column from original column with some transformed
    """
    df = (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
        .assign(id=path.stem)
        .rename_axis('cell_id')
    )
    return df


def make_train_df(json_path: str) -> pd.DataFrame:
    """ Make DataFrame of whole dataset """
    json_list = glob.glob(f'{json_path}/*.json')
    tmp_list = [read_notebook(path) for path in tqdm(json_list)]
    df = (
        pd.concat(tmp_list)
        .set_index('id', append=True)
        .swaplevel()
        .sort_index(level='id', sort_remaining=False)
    )
    return df


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data_folder from csv file like as train.csv, test.csv, val.csv
    """
    df = pd.read_csv(
        data_path,
        keep_default_na=False  #
    )
    return df


def get_ranks(base: pd.DataFrame, derived: list) -> list:
    """ return cell_id's sequence rank in unique notebook_id """
    return [base.index(d) for d in derived]


def create_word_normalizer():
    """
    Create a function that normalizes a word.
    """
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    def normalize(word):
        w = word.lower()
        w = lemmatizer.lemmatize(w)
        w = ps.stem(w)
        return w
    return normalize


def __normalize_words(titles: list) -> list:
    """
    Normalize a list of words
    1) Remove stop words
    2) Apply Porter Stemmer, Lemmatizer
    """
    stop_words = set(stopwords.words('english'))
    normalizer = create_word_normalizer()
    titles = [normalizer(t) for t in titles if t not in stop_words]
    return titles


def normalize_words(words: np.ndarray, unique=True) -> list:
    """
    Normalize a list of words
    1) Apply __normalize_word function
    2) Apply Regular Expression to remove special characters
    """
    if type(words) is str:
        words = [words]
    sep_re = r'[\s\(\){}\[\];,\.]+'
    num_re = r'\d'
    words = re.split(sep_re, ' '.join(words).lower())
    words = [w for w in words if len(w) >= 3 and not re.match(num_re, w)]
    if unique:
        words = list(set(words))
        words = set(__normalize_words(words))
    else:
        words = __normalize_words(words)
    return words


