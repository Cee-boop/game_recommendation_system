import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


def load_model():
    with open(os.environ['BASE_DIR'], 'rb') as file:
        return pickle.load(file)

# LOAD IN DATA
DF = pd.read_csv('database/game_corpus.csv')
GAME_DATABASE = DF['game_title']
GAME_SUMMARIES = DF['game_summary']
GAME_INDEX_MAP = pd.Series(DF.index, index=DF['game_title'])
GAME_TDM = load_model()


def show_homepage():
    st.title('GamesLike')
    query_title = st.selectbox('Select a game from the dropbox below:', GAME_DATABASE.tolist())
    ok = st.button('Find games!')

    if ok:
        query_vector = GAME_TDM[GAME_INDEX_MAP[query_title]]
        cos_sim_series = pd.Series(cosine_similarity(query_vector, GAME_TDM).flatten(), name='cos_sim')
        results = (
            pd.concat([GAME_DATABASE, GAME_SUMMARIES, cos_sim_series], axis=1)
            .sort_values(['cos_sim'], ascending=False)[:20]
        )

        # Create a Markdown list with each game
        rec_titles = [title for title in results['game_title'] if title != query_title][:10]
        string_markdown = ''
        for title in rec_titles:
            string_markdown += '- ' + title + '\n'

        st.markdown(string_markdown)


















