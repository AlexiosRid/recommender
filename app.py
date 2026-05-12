import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import sqlite3
import hashlib
from datetime import datetime
import plotly.express as px

# ====================== НАСТРОЙКИ ======================
st.set_page_config(page_title="MovieRec ML", layout="wide", page_icon="🎬", initial_sidebar_state="expanded")

# ====================== КАСТОМНЫЙ CSS ======================
st.markdown("""
<style>
    .movie-card {
        border-radius: 16px;
        padding: 12px;
        background: #1e2937;
        transition: all 0.3s;
        height: 100%;
    }
    .movie-card:hover {
        transform: scale(1.03);
        box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.2);
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 38px;
        font-size: 0.95rem;
        font-weight: 500;
    }
    .save-btn>button { background: linear-gradient(135deg, #6b46c1, #9f7aea); color: white; }
    .fav-btn>button { background: linear-gradient(135deg, #e11d48, #fb7185); color: white; }
    .watch-btn>button { background: linear-gradient(135deg, #0ea5e9, #38bdf8); color: white; }
    .similar-btn>button { background: linear-gradient(135deg, #14b8a6, #22d3ee); color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🎬 MovieRec ML")
st.markdown("**Гибридная рекомендательная система фильмов**")

# ====================== БАЗА ДАННЫХ ======================
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS ratings (user_id TEXT, movieId INTEGER, rating REAL, timestamp TEXT, PRIMARY KEY(user_id, movieId))''')
    c.execute('''CREATE TABLE IF NOT EXISTS favorites (user_id TEXT, movieId INTEGER, added_date TEXT, PRIMARY KEY(user_id, movieId))''')
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist (user_id TEXT, movieId INTEGER, added_date TEXT, PRIMARY KEY(user_id, movieId))''')
    conn.commit()
    conn.close()

init_db()

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username: str, password: str) -> bool:
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def login_user(username: str, password: str) -> bool:
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hash_password(password)))
    return c.fetchone() is not None

def save_rating(user_id: str, movie_id: int, rating: float):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO ratings VALUES (?, ?, ?, ?)", 
              (user_id, movie_id, rating, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def add_to_favorites(user_id: str, movie_id: int):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO favorites VALUES (?, ?, ?)", 
              (user_id, movie_id, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()

def add_to_watchlist(user_id: str, movie_id: int):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO watchlist VALUES (?, ?, ?)", 
              (user_id, movie_id, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()

def get_favorites(user_id: str) -> pd.DataFrame:
    conn = sqlite3.connect('users.db')
    df = pd.read_sql_query("SELECT movieId FROM favorites WHERE user_id=?", conn, params=(user_id,))
    conn.close()
    return df

def get_watchlist(user_id: str) -> pd.DataFrame:
    conn = sqlite3.connect('users.db')
    df = pd.read_sql_query("SELECT movieId FROM watchlist WHERE user_id=?", conn, params=(user_id,))
    conn.close()
    return df

def get_user_ratings(user_id: str) -> pd.DataFrame:
    conn = sqlite3.connect('users.db')
    df = pd.read_sql_query("SELECT * FROM ratings WHERE user_id=?", conn, params=(user_id,))
    conn.close()
    return df

# ====================== МОДЕЛИ ======================
@st.cache_resource
def build_models():
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    movies['genres_str'] = movies['genres'].str.replace('|', ' ')
    
    popular_ids = ratings['movieId'].value_counts().head(3500).index
    movies_f = movies[movies['movieId'].isin(popular_ids)].copy().reset_index(drop=True)
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=4000)
    tfidf_matrix = tfidf.fit_transform(movies_f['genres_str'])
    content_sim = cosine_similarity(tfidf_matrix, dense_output=False)
    
    user_item = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    nmf = NMF(n_components=20, random_state=42, max_iter=300)
    W = nmf.fit_transform(user_item)
    H = nmf.components_
    cf_pred = np.dot(W, H)
    
    avg_ratings = ratings.groupby('movieId')['rating'].mean().round(1)
    movies_f = movies_f.merge(avg_ratings, on='movieId', how='left')
    movies_f.rename(columns={'rating': 'avg_rating'}, inplace=True)
    
    return content_sim, movies_f, cf_pred, user_item.columns.tolist(), movies, ratings, tfidf_matrix

content_sim, movies_filtered, cf_pred, movie_ids_cf, movies, ratings, tfidf_matrix = build_models()

class HybridRecommender:
    def get_recommendations(self, user_id: str, top_n: int = 12, genre_filter=None) -> pd.DataFrame:
        user_ratings = get_user_ratings(user_id)
        if len(user_ratings) < 5:
            popular = ratings.groupby('movieId').size().nlargest(100).index
            recs = movies[movies['movieId'].isin(popular)].sample(top_n)
        else:
            user_rated_idx = movies_filtered[movies_filtered['movieId'].isin(user_ratings['movieId'])].index
            if len(user_rated_idx) == 0:
                recs = movies_filtered.sample(top_n)
            else:
                user_profile = np.mean(content_sim[user_rated_idx].toarray(), axis=0)
                hybrid_scores = []
                rated_ids = set(user_ratings['movieId'])
                for idx, row in movies_filtered.iterrows():
                    mid = row['movieId']
                    content_score = float(user_profile[idx])
                    cf_score = 0.0
                    try:
                        if int(user_id) in user_item.index:
                            user_idx = user_item.index.get_loc(int(user_id))
                            cf_score = float(cf_pred[user_idx][movie_ids_cf.index(mid)])
                    except:
                        pass
                    hybrid = 0.65 * content_score + 0.35 * cf_score
                    hybrid_scores.append((idx, hybrid))
                hybrid_scores.sort(key=lambda x: x[1], reverse=True)
                rec_idx = [idx for idx, _ in hybrid_scores if movies_filtered.iloc[idx]['movieId'] not in rated_ids][:top_n*2]
                recs = movies_filtered.iloc[rec_idx].reset_index(drop=True)
        
        if genre_filter:
            recs = recs[recs['genres'].str.contains(genre_filter, case=False, na=False)]
        return recs.head(top_n)

    def explain_recommendation(self, movie_id: int, user_id: str) -> str:
        movie = movies[movies['movieId'] == movie_id].iloc[0]
        reasons = [f"**Жанры**: {movie['genres'].replace('|', ', ')}"]
        user_ratings = get_user_ratings(user_id)
        if not user_ratings.empty:
            user_genres = movies_filtered[movies_filtered['movieId'].isin(user_ratings['movieId'])].merge(
                user_ratings, on='movieId')
            top_genres = user_genres['genres'].str.split('|').explode().value_counts().head(3).index.tolist()
            reasons.append(f"**Совпадает с любимыми жанрами**: {', '.join(top_genres)}")
        reasons.append("**Высокий гибридный score** (Content + Collaborative)")
        return "\n\n".join(reasons)

    def get_similar_movies(self, movie_id: int, top_n: int = 6) -> pd.DataFrame:
        try:
            idx = movies_filtered[movies_filtered['movieId'] == movie_id].index[0]
            sim_scores = list(enumerate(content_sim[idx].toarray()[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            similar_idx = [i for i, _ in sim_scores[1:top_n+1]]
            return movies_filtered.iloc[similar_idx].reset_index(drop=True)
        except:
            return pd.DataFrame()

recommender = HybridRecommender()

# ====================== АВТОРИЗАЦИЯ ======================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.current_recs = None
    st.session_state.similar_to = None

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("Добро пожаловать в MovieRec ML")
        tab1, tab2 = st.tabs(["🔑 Вход", "📝 Регистрация"])
        with tab1:
            un = st.text_input("Имя пользователя", key="login_user")
            pw = st.text_input("Пароль", type="password", key="login_pass")
            if st.button("Войти", use_container_width=True):
                if login_user(un, pw):
                    st.session_state.logged_in = True
                    st.session_state.username = un
                    st.success("Успешный вход!")
                    st.rerun()
                else:
                    st.error("Неверные данные")
        with tab2:
            un2 = st.text_input("Имя пользователя", key="reg_user")
            pw2 = st.text_input("Пароль", type="password", key="reg_pass")
            if st.button("Зарегистрироваться", use_container_width=True):
                if register_user(un2, pw2):
                    st.success("Аккаунт создан! Теперь войдите.")
                else:
                    st.error("Пользователь уже существует")
    st.stop()

# ====================== ОСНОВНОЙ ИНТЕРФЕЙС ======================
st.sidebar.success(f"👤 {st.session_state.username}")
if st.sidebar.button("🚪 Выйти"):
    st.session_state.logged_in = False
    st.rerun()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Рекомендации", "⭐ Оценить", "❤️ Избранное", 
    "📋 Смотреть позже", "📊 Мои оценки", "📈 Аналитика"
])

with tab1:
    st.subheader("🎯 Гибридные персональные рекомендации")
    
    colf1, colf2 = st.columns([3, 1])
    with colf1:
        all_genres = ["Все жанры"] + sorted(set('|'.join(movies['genres'].dropna()).split('|')))
        genre_filter = st.selectbox("Фильтр по жанру", all_genres)
    with colf2:
        num_recs = st.slider("Количество рекомендаций", 6, 18, 12)
    
    if st.button("🔄 Обновить рекомендации", type="primary", use_container_width=True):
        with st.spinner("Гибридная модель работает..."):
            genre = None if genre_filter == "Все жанры" else genre_filter
            recs = recommender.get_recommendations(st.session_state.username, num_recs, genre)
            st.session_state.current_recs = recs
            st.session_state.similar_to = None

    if 'current_recs' in st.session_state and st.session_state.current_recs is not None:
        recs = st.session_state.current_recs
        cols = st.columns(3)
        for i, row in recs.iterrows():
            with cols[i % 3]:
                with st.container(border=True):
                    poster_url = f"https://picsum.photos/id/{(row['movieId'] % 900) + 100}/280/400"
                    st.image(poster_url, use_container_width=True)
                    
                    st.markdown(f"**{row['title']}**")
                    st.caption(row['genres'])
                    if 'avg_rating' in row and pd.notna(row['avg_rating']):
                        st.caption(f"⭐ Средний: {row['avg_rating']}")
                    
                    rating = st.slider("Ваша оценка", 0.5, 5.0, 4.0, 0.5, key=f"sl_{row['movieId']}_{i}")
                    
                    btn_cols = st.columns(4)
                    with btn_cols[0]:
                        st.markdown('<div class="save-btn">', unsafe_allow_html=True)
                        if st.button("💾", key=f"save_{row['movieId']}_{i}"):
                            save_rating(st.session_state.username, int(row['movieId']), rating)
                            st.success("✅")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with btn_cols[1]:
                        st.markdown('<div class="fav-btn">', unsafe_allow_html=True)
                        if st.button("❤️", key=f"fav_{row['movieId']}_{i}"):
                            add_to_favorites(st.session_state.username, int(row['movieId']))
                            st.success("❤️")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with btn_cols[2]:
                        st.markdown('<div class="watch-btn">', unsafe_allow_html=True)
                        if st.button("📋", key=f"wl_{row['movieId']}_{i}"):
                            add_to_watchlist(st.session_state.username, int(row['movieId']))
                            st.success("📋")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with btn_cols[3]:
                        st.markdown('<div class="similar-btn">', unsafe_allow_html=True)
                        if st.button("🔍", key=f"sim_{row['movieId']}_{i}"):
                            st.session_state.similar_to = int(row['movieId'])
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with st.expander("🔍 Почему рекомендуем?"):
                        exp = recommender.explain_recommendation(int(row['movieId']), st.session_state.username)
                        st.markdown(exp)

    if st.session_state.get('similar_to'):
        st.subheader("🔍 Похожие фильмы")
        similar = recommender.get_similar_movies(st.session_state.similar_to, 6)
        sim_cols = st.columns(3)
        for j, srow in similar.iterrows():
            with sim_cols[j % 3]:
                with st.container(border=True):
                    st.image(f"https://picsum.photos/id/{(srow['movieId'] % 900) + 100}/280/400", use_container_width=True)
                    st.markdown(f"**{srow['title']}**")
                    st.caption(srow['genres'])
        if st.button("Закрыть похожие"):
            st.session_state.similar_to = None
            st.rerun()

with tab2:
    st.subheader("🔍 Поиск и оценка фильмов")
    search = st.text_input("Поиск фильма", "")
    if search:
        results = movies[movies['title'].str.contains(search, case=False, na=False)].head(12)
        cols = st.columns(3)
        for _, row in results.iterrows():
            with cols[_ % 3]:
                with st.container(border=True):
                    st.image(f"https://picsum.photos/id/{(row['movieId'] % 900) + 100}/280/400", use_container_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.caption(row['genres'])
                    rating = st.slider("Оценка", 0.5, 5.0, 4.0, 0.5, key=f"sl2_{row['movieId']}")
                    if st.button("💾 Сохранить", key=f"sv2_{row['movieId']}"):
                        save_rating(st.session_state.username, int(row['movieId']), rating)
                        st.success("✅ Сохранено")

with tab3:
    st.subheader("❤️ Избранное")
    favs = get_favorites(st.session_state.username)
    if not favs.empty:
        fav_movies = movies[movies['movieId'].isin(favs['movieId'])].reset_index(drop=True)
        cols = st.columns(3)
        for _, row in fav_movies.iterrows():
            with cols[_ % 3]:
                with st.container(border=True):
                    st.image(f"https://picsum.photos/id/{(row['movieId'] % 900) + 100}/280/400", use_container_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.caption(row['genres'])
    else:
        st.info("В избранном пока пусто")

with tab4:
    st.subheader("📋 Смотреть позже")
    wl = get_watchlist(st.session_state.username)
    if not wl.empty:
        wl_movies = movies[movies['movieId'].isin(wl['movieId'])].reset_index(drop=True)
        cols = st.columns(3)
        for _, row in wl_movies.iterrows():
            with cols[_ % 3]:
                with st.container(border=True):
                    st.image(f"https://picsum.photos/id/{(row['movieId'] % 900) + 100}/280/400", use_container_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.caption(row['genres'])
    else:
        st.info("Список «Смотреть позже» пуст")

with tab5:
    st.subheader("📊 Мои оценки")
    ur = get_user_ratings(st.session_state.username)
    if not ur.empty:
        full = ur.merge(movies[['movieId', 'title', 'genres']], on='movieId')
        st.dataframe(full[['title', 'genres', 'rating', 'timestamp']].sort_values('rating', ascending=False), 
                    use_container_width=True)
    else:
        st.info("Вы ещё не оценили ни одного фильма")

with tab6:
    st.subheader("📈 Аналитика")
    ur = get_user_ratings(st.session_state.username)
    if not ur.empty:
        full = ur.merge(movies[['movieId', 'title', 'genres']], on='movieId')
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Оценено", len(ur))
        c2.metric("Средняя оценка", round(ur['rating'].mean(), 2))
        c3.metric("5.0", len(ur[ur['rating'] == 5.0]))
        c4.metric("В Watchlist", len(get_watchlist(st.session_state.username)))
        
        st.subheader("Распределение оценок")
        st.bar_chart(ur['rating'].value_counts().sort_index())
        
        st.subheader("🎭 Анализ жанров")
        full['genre_list'] = full['genres'].str.split('|')
        exploded = full.explode('genre_list')
        genre_stats = exploded.groupby('genre_list').agg(
            count=('rating', 'size'),
            mean_rating=('rating', 'mean')
        ).round(2)
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(px.bar(genre_stats.nlargest(10, 'count'), x='count', 
                                 y=genre_stats.nlargest(10, 'count').index, orientation='h',
                                 title="Топ-10 жанров по количеству"), use_container_width=True)
        with col_g2:
            st.plotly_chart(px.bar(genre_stats.nlargest(10, 'mean_rating'), x='mean_rating', 
                                 y=genre_stats.nlargest(10, 'mean_rating').index, orientation='h',
                                 title="Топ-10 жанров по средней оценке"), use_container_width=True)
        
        st.subheader("📊 Круговые диаграммы")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.plotly_chart(px.pie(genre_stats.reset_index(), names='genre_list', values='count',
                                  title="Доля оценок по жанрам", hole=0.4), use_container_width=True)
        with col_p2:
            st.plotly_chart(px.pie(genre_stats.reset_index(), names='genre_list', values='mean_rating',
                                  title="Средняя оценка по жанрам", hole=0.4), use_container_width=True)
    else:
        st.info("Оцените фильмы для отображения аналитики")

st.caption("Курсовая работа | Гибридная рекомендательная система | Streamlit + scikit-learn + Plotly")