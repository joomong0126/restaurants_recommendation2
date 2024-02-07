import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle
from PyQt5.QtCore import QStringListModel

form_window = uic.loadUiType('./restaurants_recommendation2.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.Tfidf_matrix = mmread('./models/Tfidf_naver_review.mtx').tocsr()
        with open('./models/tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./models/word2vec_naver_review.model')
        self.df_reviews = pd.read_csv('./cleaned_naver.csv')
        self.titles = list(self.df_reviews['names'])
        self.titles.sort()
        for title in self.titles:
            self.comboBox.addItem(title)

        model = QStringListModel()
        model.setStringList(self.titles)
        completer = QCompleter()
        completer.setModel(model)
        self.le_keyword.setCompleter(completer)

        self.comboBox.currentIndexChanged.connect(self.combobox_slot)
        self.btn_recommendation.clicked.connect(self.btn_slot)

    def btn_slot(self):
        key_words = self.le_keyword.text().split()  # 여러 키워드를 띄어쓰기로 구분하여 리스트로 저장
        recommendations, keyword_counts = self.find_common_restaurants(key_words)

        if recommendations:
            self.lbl_recommendation.setText('\n'.join(recommendations))
        else:
            self.lbl_recommendation.setText('일치하는 음식점이 없습니다.')

        # 터미널에 키워드 갯수 출력
        for i, key_word in enumerate(key_words):
            print(f"{i+1}번째 키워드 '{key_word}'의 리뷰에서 찾은 유사한 단어 수: {keyword_counts[i]}")

    def combobox_slot(self):
        title = self.comboBox.currentText()
        print(title)
        recommendation = self.recommendation_by_movie_title(title)
        print('debug01')
        self.lbl_recommendation.setText(recommendation)
        print('debug02')

    def recommendation_by_keyword(self, key_word):
        try:
            sim_word = self.embedding_model.wv.most_similar(key_word, topn=50)
        except:
            self.lbl_recommendation.setText('제가 모르는 단어에요 ㅠㅠ')
            return [], 0

        keyword_restaurants = []
        keyword_counts = {}  # 음식점별 키워드 발견 횟수 저장

        for word, _ in sim_word: # 주어진 키워드와 유사한 단어를 찾아냅니다.
            # 키워드와 유사한 단어가 들어간 리뷰를 찾아서 해당 음식점들을 리스트에 추가
            restaurants = self.df_reviews[self.df_reviews['reviews'].str.contains(word)]['names'].tolist()
            # 리스트에는 각 유사한 단어가 나타난 리뷰에서 얻은 음식점 목록이 저장됩니다.
            keyword_restaurants.extend(restaurants)

            # 유사한 단어들을 터미널에서 출력
            print(f"키워드 '{key_word}'와 유사한 단어: {word}")


        # 음식점 리스트에서 키워드가 많이 언급된 순서대로 정렬
        sorted_restaurants = sorted(keyword_counts.keys(), key=lambda x: keyword_counts[x], reverse=True)
        # 음식점 리스트를 키워드가 많이 언급된 순서대로 정렬하고, 상위 10개의 음식점을 선택하여 출력합니다.

        # 상위 10개 추천
        recommendation_restaurants = sorted_restaurants[:10]
        recommendation_counts = [keyword_counts[restaurant] for restaurant in recommendation_restaurants]
        recommendation = [f"{restaurant} (발견 횟수: {count})" for restaurant, count in
                          zip(recommendation_restaurants, recommendation_counts)]

        # 키워드와 관련된 음식점들을 즉시 출력
        print(f"\n키워드 '{key_word}'와 관련된 음식점 추천:")
        print(recommendation)

        return recommendation, len(recommendation_restaurants)

    def recommendation_by_movie_title(self, title):
        movie_idx = self.df_reviews[self.df_reviews['names'] == title].index[0]
        cosine_sim = linear_kernel(self.Tfidf_matrix[movie_idx], self.Tfidf_matrix)
        recommendation = self.getRecommendation(cosine_sim)
        recommendation = [f"{movie}" for movie in recommendation]
        return recommendation

    def getRecommendation(self, cosine_sim):
        simScore = list(enumerate(cosine_sim[-1]))
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
        simScore = simScore[:11]
        movieIdx = [i[0] for i in simScore]
        recmovieList = self.df_reviews.iloc[movieIdx, 0]
        return recmovieList[1:11]

    def find_common_restaurants(self, key_words):
        common_restaurants = []
        keyword_counts = []

        for key_word in key_words:
            # 키워드와 유사한 단어가 들어간 리뷰를 찾아서 해당 음식점들을 리스트에 추가
            restaurants = self.df_reviews[self.df_reviews['reviews'].str.contains(key_word)]['names'].tolist()
            keyword_counts.append(len(restaurants))

            if not common_restaurants:
                common_restaurants = restaurants
            else:
                # 현재 찾은 음식점들과 새로 찾은 음식점들 중복 체크
                common_restaurants = list(set(common_restaurants).intersection(restaurants))

        return common_restaurants[:10], keyword_counts


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())

