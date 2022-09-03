import pandas as pd
from textblob import TextBlob, Word

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pymorphy2


nltk.download('all')

greetings = {'здравствуйте', 'добрый день', 'добрый вечер', 'доброе утро'}
farewells = {'до свидания', 'всего доброго', 'всего хорошего'}
intro_words = {'зовут', 'имя', 'это'}

f_data = 'test_data.csv'


def preprocess_data(f_data=f_data):
    data = pd.read_csv(f_data)

    m_data = data[data['role']=='manager']
    dlg_count = pd.unique(m_data['dlg_id'])

    data_txt = m_data.groupby(['dlg_id'])['text'].transform(lambda x: '. '.join(x)).drop_duplicates().reset_index()
    data_txt['text'] = data_txt['text'].transform(lambda x: x.lower())

    bls = [TextBlob(data_txt['text'][i]) for i in dlg_count]
    return data_txt, bls


def find_greets_and_farws(bls):
    greet_replica = {}
    farw_replica = {}

    for i_idx, i in enumerate(bls):
      for j in i.sentences:
        greet = [k for k in greetings if k in j]
        farw = [k for k in farewells if k in j]
        if greet:
          greet_replica[i_idx] = j
        if farw:
          farw_replica[i_idx] = j

    return greet_replica, farw_replica


def check_greet_and_farw(greet_replica, farw_replica, bls):
    set_gf = set(greet_replica.keys()) & set(farw_replica.keys())
    return {i: i in set_gf for i in range(len(bls))}


def find_manag_intro_and_sents_comp_in(data_txt, bls):
    p_thresh = 0.5
    morph = pymorphy2.MorphAnalyzer()

    managers = {}
    intro_replica = {}
    company_sentences = {}
    names =[]

    for speech in data_txt["text"]:
        names_ = []
        for idx, word in enumerate(nltk.word_tokenize(speech)):
            # print(word)
            for p in morph.parse(word):
                if 'Name' in p.tag and p.score >= p_thresh:
                    names_.append(word)
        names.append(names_)

    for i_idx, i in enumerate(bls):
      for j in i.ngrams():  # по умолчанию 3 слова
        man_name = list(set(j) & set(names[i_idx]))
        is_intro_word = list(set(j) & intro_words)
        if is_intro_word and man_name:
          managers[i_idx] = man_name[0]
      for j in i.sentences:
        if i_idx in intro_replica.keys() and i_idx in company_sentences.keys():
          break
        if i_idx in managers.keys() and managers[i_idx] in j:
          intro_replica[i_idx] = j
        if 'компания' in j:
          company_sentences[i_idx] = j

    return managers, intro_replica, company_sentences


def find_comp_names(company_sentences):
    stop_words = stopwords.words('russian')
    morph = pymorphy2.MorphAnalyzer()
    sp_parts_in_company_name = ['NOUN', 'ADJF']

    sentences = {idx: nltk.word_tokenize(str(i)[:-1]) for idx, i in company_sentences.items()}
    sp_parts = {}
    companies = {}

    for idx, i in sentences.items():
      tokens = [s for s in i if s not in stop_words]
      sp_parts[idx] = {t: str(morph.parse(t)[0].tag).split(',')[0] for t in tokens}
      comp_name = ''
      k = list(sp_parts[idx].keys())
      v = list(sp_parts[idx].values())
      co_idx = k.index('компания')
      k = k[co_idx + 1:]
      v = v[co_idx + 1:]
      for c in zip(k, v):
        if c[1] in sp_parts_in_company_name:
          comp_name = comp_name + c[0] + ' '
        else:
          break

      companies[idx] = comp_name[:-1]

    return companies


data_txt, bls = preprocess_data()
greet_replica, farw_replica = find_greets_and_farws(bls)
managers_with_g_f = check_greet_and_farw(greet_replica, farw_replica, bls)
managers, intro_replica, company_sentences = find_manag_intro_and_sents_comp_in(data_txt, bls)
companies = find_comp_names(company_sentences)

print('Greetings:\n', greet_replica)
print('Farwells:\n', farw_replica, '\n')
print('Managers were greeting and farwelling:\n', managers_with_g_f, '\n')
print('Managers names: ', managers, '\n')
print('Managers introducing selves replicas:\n', intro_replica, '\n')
print('Companies names:\n', companies)
