from word_counter import WordCounter
import numpy as np


def test_word_counter_fit_always_succeeds():
    wc = WordCounter()
    anything = None 
    wc2 = wc.fit(anything)
    assert wc == wc2


def test_word_counter_transform_success():
    wc = WordCounter()
    X = ["The first sentence", "and the second sentence. This time     with some 'special' characters", "", "a"] 
    Xt = wc.transform(X)
    assert Xt.shape == (4,1)
    assert Xt[0].tolist() == [3, 10, 0, 1]
    X = [] 
    Xt = wc.transform(X)
    assert Xt.shape == (0,0)

    
def test_word_counter_transform_fail():
    wc = WordCounter()
    try:
        X = ["abc", 3, "def"] 
        Xt = wc.transform(X)
        assert False
    except AttributeError: 
        pass
    try:
        X = ["abc", None, "def"] 
        Xt = wc.transform(X)
        assert False
    except AttributeError: 
        pass
    try:
        X = ["abc", np.NaN, "def"] 
        Xt = wc.transform(X)
        assert False
    except AttributeError: 
        pass
    
