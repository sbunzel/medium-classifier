import pandas as pd
import numpy as np
from numpy import ndarray
from collections import namedtuple
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from pathlib import Path
import copy
import re
import operator


def save_pickle(df: pd.DataFrame, out:Path) -> None:
    m = df.loc[0, "method"]
    p = Path(out / "raw" / re.sub(" ", "_", m[:-3]))
    if not p.is_dir(): p.mkdir(parents=True)
    f = p / f"{re.sub(' ', '_', m)}.pkl"
    df.to_pickle(f)
    print(f"Saved results to file at: {str(f)}")
    

def save_html(df:pd.DataFrame, name:str, out:Path, **kwargs) -> None:
    html = (df
            .to_html(**kwargs)
            .replace('<tr>', '<tr align="center">')
            .replace('<tr style="text-align: right;">', '<tr style="text-align: center;">')
            .replace('border="1"', 'border="2"')
            .replace('<th>', '<th><center>'))
    with open(out / f"{name}.html", "w") as f:
        f.write(html)
        
        
def read_results(folder: Path, pattern: str) -> List:
    results = []
    for f in folder.rglob(pattern):
        df = pd.read_pickle(f)
        df["base_method"] = f.parent.name
        results.append(df)
    return results
        
        
class CustomEvaluator:
    
    def __init__(self, target_precision:float=0.95, pos_label:int=1):
        self.target_precision = target_precision
        self.pos_label = pos_label
        
    def score(self, y_true:ndarray, probas_pred:ndarray, return_res:bool=False):
        prs, rcs, ths = metrics.precision_recall_curve(y_true, probas_pred, pos_label=self.pos_label)
    
        auc = metrics.roc_auc_score(y_true, probas_pred)
        print(f"AUC SCORE: {auc:.2f}")
        results = pd.DataFrame({"precision": prs[:-1], "recall": rcs[:-1], "threshold": ths})\
                    .sort_values(by=["precision", "recall"], ascending=[False, False])
        
        if return_res:
            return auc
        elif np.max(results.precision) > self.target_precision:
            print(results[results.precision >= self.target_precision])
        else:
            print(results.head(3))
        

ScoredClf = namedtuple("ScoredClf", [
    "clf",
    "train_auc",
    "oob_auc"
])


def fit_ensemble(model, s:StratifiedShuffleSplit, X:ndarray, y:ndarray, print_progress:bool=False, **kwargs) -> List:
    """Fit a model on different subsets of the training set and collect the results

    Arguments:
    m - a model object implementing `fit` and `predict_proba` or a tuple specifying a keras model architecture
    s - an object of class sklearn.model_selection.StratifiedShuffleSplit, i.e. an iterator of random, stratified splits
    X - numpy array of training texts
    y - numpy array of training labels
    **kwargs - keyword arguments to pass to the model when calling fit

    Returns:
    fitted_clfs - a list of named tuples collecting the scored classifiers as well as their training and out of bag AUCs
    """
    
    from keras.engine.training import Model as keras_model
    from keras.models import Model

    fitted_clfs = []

    for i, split in enumerate(s.split(X, y)):
        i_train = split[0]
        i_test = split[1]
        
        if print_progress:
            print("#######################################")
            print("Training model number  ", i+1)
            print("#######################################")
            print("")
            
        if isinstance(model, tuple):
            m = Model(inputs=model[0], outputs=model[1])
            m.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        else:
            m = model

        m.fit(X[i_train], y[i_train], **kwargs)
        fitted_clf = copy.deepcopy(m)

        if hasattr(m, "predict_proba"):
            p1_train = fitted_clf.predict_proba(X[i_train])[:, 1]
            p1_oob = fitted_clf.predict_proba(X[i_test])[:, 1]
        elif isinstance(m, keras_model):
            p1_train = fitted_clf.predict(X[i_train])
            p1_oob = fitted_clf.predict(X[i_test])

        train_auc = metrics.roc_auc_score(y[i_train], p1_train)
        oob_auc = metrics.roc_auc_score(y[i_test], p1_oob)
        fitted_clfs.append(ScoredClf(fitted_clf, train_auc, oob_auc))

        if print_progress:
            print(f"TRAIN AUC: {train_auc:.2f}")
            print("")
            print(f"OOB AUC: {oob_auc:.2f}")
            print("")

    return fitted_clfs


def evaluate_ensemble(fitted:List, eval:CustomEvaluator, X_test:ndarray,
                      y_test:ndarray, return_res:bool=False, method:str="default") -> pd.DataFrame:
    """Evaluate the performance of a set of classifiers trained on different subsets of the training set
    
    Arguments:
    fitted - list of named tuples containing fitted models as well as their train and out of bag AUC
    eval - object of class CustomEvaluator used to evaluate the performance on the hold out set
    X_test - numpy array of the texts for the hold out set for final evaluation
    y_test - numpy array of labels for the hold out set for final evaluation
    
    Return:
    pd.DataFrame - if requested, return pandas dataframe summarizing the results
    """
    
    from keras.engine.training import Model as keras_model
    
    if hasattr(fitted[0], "clf"):
        train_scores = [m.train_auc for m in fitted]
        oob_scores = [m.oob_auc for m in fitted]
        
        if hasattr(fitted[0].clf, "predict_proba"):
            preds_test = np.array([m.clf.predict_proba(X_test)[:, 1] for m in fitted])
        elif isinstance(fitted[0].clf, keras_model):
            preds_test = np.array([m.clf.predict(X_test) for m in fitted])
        
        print(f"Mean Train AUC: {np.mean(train_scores):.2f} (+/- {np.std(train_scores):.2f})")
        print(f"Mean OOB AUC: {np.mean(oob_scores):.2f} (+/- {np.std(oob_scores):.2f})")
        print("")
        
    else: preds_test = np.array([m.predict_proba(X_test)[:, 1] for m in fitted])
    print("Performance on hold out set:")
    if return_res:
        test_auc = eval.score(y_test, preds_test.mean(axis=0), return_res)
        return pd.DataFrame({"method": method,
                             "mean train auc" : np.mean(train_scores),
                             "mean cv auc" : np.mean(oob_scores),
                             "mean test auc" : test_auc}, index=[0])
    else:
        eval.score(y_test, preds_test.mean(axis=0))
    

    
def build_vocab(sentences):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def check_coverage(vocab, embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in vocab:
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print("Found matches for {:.2%} of vocab".format(len(a) / len(vocab)))
    print("Found matches for  {:.2%} of all text".format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

    
def clean_apostrophe(txt):
    
    txt = re.sub("it’s", "it is", txt)
    txt = re.sub("don’t", "do not", txt)
    txt = re.sub("i’m", "i am", txt)
    txt = re.sub("we’ll", "we will", txt)
    txt = re.sub("let’s", "let us", txt)
    txt = re.sub("we’re", "we are", txt)
    txt = re.sub("there’s", "there is", txt)
    txt = re.sub("didn’t", "did not", txt)
    txt = re.sub("you’re", "you are", txt)
    txt = re.sub("i’ve", "i have", txt)
    txt = re.sub("isn’t", "is not", txt)
    txt = re.sub("here’s", "here is", txt)
    txt = re.sub("that’s", "that is", txt)
    txt = re.sub("we’ve", "we have", txt)
    txt = re.sub("can’t", "cannot", txt)
    txt = re.sub("won’t", "will not", txt)
    txt = re.sub("you’ll", "you will", txt)
    txt = re.sub("aren’t", "are not", txt)
    txt = re.sub("what’s", "what is", txt)
    txt = re.sub("wasn’t", "was not", txt)
    txt = re.sub("doesn’t", "does not", txt)
    txt = re.sub("havent", "have not", txt)
    txt = re.sub("youve", "you have", txt)
    txt = re.sub("theyre", "they are", txt)
    txt = re.sub("youd", "you would", txt)
    txt = re.sub("wouldnt", "would not", txt)
    txt = re.sub("couldnt", "could not", txt)

    return txt


def fix_punctuation(txt):
    
    txt = txt.replace(",", " ,")
    txt = txt.replace(".", " .")
    
    return txt


def remove_punctuation(txt):
    
    for punct in '?!.,-"#&$%\'()*+-/:;<=>@[\\]^_`{|}~—' + '“”’':
        txt = txt.replace(punct, "")
        
    txt = txt.replace("\n", " ")
    
    return txt


def fix_specific(txt):
    
    for t in ["neural network", "neural networks", "neuralnetworks", "neural net", "neural nets"]:
        txt = txt.replace(t, "neuralnetwork")
        
    for t in ["rnn", "rnns", "lstm", "recurrent"]:
        txt = txt.replace(t, "rnn")
        
    for t in ["keras"]:
        txt = txt.replace(t, "tensorflow")
        
    for t in ["convolution", "cnn", "rcnn"]:
        txt = txt.replace(t, "cnn")
        
    for mooc in ["udacity", "moocs", "udemy", "nanodegree", "cs231n", "coursera"]:
        txt = txt.replace(mooc, "mooc")
        
    txt = txt.replace("neurons", "neuron")
    txt = txt.replace("gpus", "gpu")
    txt = txt.replace("gradient descent", "gradientdescent")
    txt = txt.replace("google", "googles")
        
    
    return txt


def clean_numbers(x):

    x = re.sub("[0-9]{5,}", "#####", x)
    #x = re.sub("[0-9]{4}", "####", x)
    x = re.sub("[0-9]{3}", "###", x)
    x = re.sub("[0-9]{2}", "##", x)
    x = re.sub("[0-9]{1}", "##", x)
    
    return x