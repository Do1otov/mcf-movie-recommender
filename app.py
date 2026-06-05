import streamlit as st
import torch
import pandas as pd
import numpy as np
from src.config import CFG
from src.models import MCF_Full
from src.dataset import load_and_split


@st.cache_resource
def load_everything():
    train_df, val_df, test_df, n_users, n_items, item_ids = load_and_split(
        CFG.data_path
    )
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    model = MCF_Full(n_users, n_items)
    state = torch.load(
        f"{CFG.checkpoint_dir}/mcf_full_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state)
    model.eval()

    movies = pd.read_csv("data/movies.csv")

    idx_to_item = {v: k for k, v in item_ids.items()}

    genre_names = {
        0: "Боевик",
        1: "Приключения",
        2: "Мультфильм",
        3: "Детский",
        4: "Комедия",
        5: "Криминал",
        6: "Документальный",
        7: "Драма",
        8: "Фэнтези",
        9: "Фильм-нуар",
        10: "Ужасы",
        11: "IMAX",
        12: "Мюзикл",
        13: "Детектив",
        14: "Мелодрама",
        15: "Фантастика",
        16: "Триллер",
        17: "Неизвестно",
        18: "Военный",
        19: "Вестерн",
    }

    return model, full_df, movies, n_users, n_items, item_ids, idx_to_item, genre_names


@st.cache_data
def get_metadata_tensor(
    movie_ids: list, movies_df: pd.DataFrame, item_ids: dict
) -> tuple:
    """Формируем тензор метаданных для списка movie_id."""
    import re

    rows = []
    valid_ids = []
    for mid in movie_ids:
        row = movies_df[movies_df["movie_id"] == mid]
        if row.empty or mid not in item_ids:
            continue
        title = row["movie_title"].values[0]
        year_match = re.search(r"\((\d{4})\)", str(title))
        year = int(year_match.group(1)) if year_match else 1990
        year_norm = round((year - 1919) / (2000 - 1919), 4)

        genre_str = str(row["movie_genres"].values[0])
        genre_vec = [0] * 20
        for g in genre_str.split(","):
            try:
                idx = int(g.strip())
                if 0 <= idx < 20:
                    genre_vec[idx] = 1
            except ValueError:
                pass

        rows.append([year_norm] + genre_vec)
        valid_ids.append(mid)

    return torch.tensor(rows, dtype=torch.float32), valid_ids


def get_user_genres(user_history: pd.DataFrame, movies_df: pd.DataFrame) -> str:
    """Определяем топ-жанры пользователя по его истории."""
    genre_counts = [0] * 20
    genre_names_map = {
        0: "Боевик",
        1: "Приключения",
        2: "Мультфильм",
        3: "Детский",
        4: "Комедия",
        5: "Криминал",
        6: "Документальный",
        7: "Драма",
        8: "Фэнтези",
        9: "Фильм-нуар",
        10: "Ужасы",
        11: "IMAX",
        12: "Мюзикл",
        13: "Детектив",
        14: "Мелодрама",
        15: "Фантастика",
        16: "Триллер",
        17: "Неизвестно",
        18: "Военный",
        19: "Вестерн",
    }
    for mid in user_history["movie_id"]:
        row = movies_df[movies_df["movie_id"] == mid]
        if row.empty:
            continue
        genre_str = str(row["movie_genres"].values[0])
        for g in genre_str.split(","):
            try:
                idx = int(g.strip())
                if 0 <= idx < 20:
                    genre_counts[idx] += 1
            except ValueError:
                pass
    top = sorted(range(20), key=lambda i: genre_counts[i], reverse=True)[:3]
    return ", ".join(genre_names_map[i] for i in top if genre_counts[i] > 0)


@torch.no_grad()
def recommend_for_user(
    model,
    user_idx_val: int,
    watched_ids: set,
    all_item_idxs: list,
    all_movie_ids: list,
    metadata_tensor: torch.Tensor,
    top_n: int = 10,
) -> tuple:
    """Получаем топ-N рекомендаций для пользователя."""
    candidates = [
        (iidx, mid)
        for iidx, mid in zip(all_item_idxs, all_movie_ids)
        if mid not in watched_ids
    ]
    if not candidates:
        return [], []

    c_item_idxs, c_movie_ids = zip(*candidates)
    n = len(c_item_idxs)

    user_tensor = torch.tensor([user_idx_val] * n, dtype=torch.long)
    item_tensor = torch.tensor(list(c_item_idxs), dtype=torch.long)

    meta_rows = []
    for mid in c_movie_ids:
        orig_idx = all_movie_ids.index(mid)
        meta_rows.append(metadata_tensor[orig_idx].unsqueeze(0))
    meta_tensor = torch.cat(meta_rows, dim=0)

    overall, criteria_scores, weights = model(user_tensor, item_tensor, meta_tensor)
    overall_np = overall.numpy()
    criteria_np = criteria_scores.numpy()
    weights_np = weights.numpy()

    top_idx = np.argsort(overall_np)[::-1][:top_n]

    recs = []
    for i in top_idx:
        recs.append(
            {
                "movie_id": c_movie_ids[i],
                "score": float(overall_np[i]),
                "plot": float(criteria_np[i][0]),
                "visual": float(criteria_np[i][1]),
                "acting": float(criteria_np[i][2]),
                "emotion": float(criteria_np[i][3]),
            }
        )

    avg_weights = weights_np[top_idx].mean(axis=0)
    return recs, avg_weights


st.set_page_config(
    page_title="Многокритериальные рекомендации фильмов",
    page_icon="🎬",
    layout="wide",
)

st.title("Многокритериальная рекомендательная система фильмов")
st.caption(
    "Дипломная работа · Neural CF с динамическими персональными весами критериев"
)

model, full_df, movies_df, n_users, n_items, item_ids, idx_to_item, genre_names = (
    load_everything()
)

all_movie_ids = list(item_ids.keys())
all_item_idxs = [item_ids[mid] for mid in all_movie_ids]

meta_tensor, valid_movie_ids = get_metadata_tensor(all_movie_ids, movies_df, item_ids)
valid_item_idxs = [item_ids[mid] for mid in valid_movie_ids]

st.sidebar.header("Настройки")
mode = st.sidebar.radio(
    "Режим",
    ["Существующий пользователь", "Новый пользователь"],
)
top_n = st.sidebar.slider("Топ-N рекомендаций", 5, 20, 10)


if mode == "Существующий пользователь":
    st.header("Рекомендации для пользователя из датасета")

    user_ids_list = sorted(full_df["user_id"].unique().tolist())
    selected_user = st.selectbox("Выбери пользователя", user_ids_list)

    user_history = full_df[full_df["user_id"] == selected_user].copy()
    user_history = user_history.merge(
        movies_df[["movie_id", "movie_title"]], on="movie_id", how="left"
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(f"История просмотров ({len(user_history)} фильмов)")
        top_genres = get_user_genres(user_history, movies_df)
        st.caption(f"Любимые жанры: **{top_genres}**")

        st.metric("Средняя оценка", f"{user_history['overall'].mean():.2f} / 5")

        history_display = user_history[
            ["movie_title", "overall", "plot", "visual", "acting", "emotion"]
        ].copy()
        history_display.columns = [
            "Фильм",
            "Overall",
            "Сюжет",
            "Визуал",
            "Актёры",
            "Атмосфера",
        ]
        history_display = history_display.sort_values("Overall", ascending=False)
        st.dataframe(history_display, use_container_width=True, height=400)

    with col2:
        st.subheader(f"Топ-{top_n} рекомендаций")

        if st.button("Получить рекомендации", type="primary"):
            user_idx_val = user_history["user_idx"].iloc[0]
            watched_ids = set(user_history["movie_id"].tolist())

            with st.spinner("Вычисляем рекомендации..."):
                recs, avg_weights = recommend_for_user(
                    model,
                    user_idx_val,
                    watched_ids,
                    valid_item_idxs,
                    valid_movie_ids,
                    meta_tensor,
                    top_n,
                )

            if not recs:
                st.warning("Не удалось найти рекомендации.")
            else:
                st.subheader("Персональный профиль весов критериев")
                criteria_labels = ["Сюжет", "Визуал", "Актёры", "Атмосфера"]
                weights_df = pd.DataFrame(
                    {
                        "Критерий": criteria_labels,
                        "Вес": avg_weights,
                    }
                )
                st.bar_chart(weights_df.set_index("Критерий"))

                recs_enriched = []
                for r in recs:
                    movie_row = movies_df[movies_df["movie_id"] == r["movie_id"]]
                    title = (
                        movie_row["movie_title"].values[0]
                        if not movie_row.empty
                        else str(r["movie_id"])
                    )

                    genre_str = (
                        str(movie_row["movie_genres"].values[0])
                        if not movie_row.empty
                        else ""
                    )
                    genres = []
                    for g in genre_str.split(","):
                        try:
                            genres.append(genre_names.get(int(g.strip()), ""))
                        except ValueError:
                            pass

                    recs_enriched.append(
                        {
                            "Фильм": title,
                            "Жанры": ", ".join(genres[:3]),
                            "Прогноз": round(r["score"], 2),
                            "Сюжет": round(r["plot"], 2),
                            "Визуал": round(r["visual"], 2),
                            "Актёры": round(r["acting"], 2),
                            "Атмосфера": round(r["emotion"], 2),
                        }
                    )

                recs_df = pd.DataFrame(recs_enriched)
                st.dataframe(
                    recs_df,
                    use_container_width=True,
                    column_config={
                        "Прогноз": st.column_config.ProgressColumn(
                            "Прогноз", min_value=1, max_value=5, format="%.2f"
                        ),
                    },
                )


else:
    st.header("Рекомендации для нового пользователя")
    st.info(
        "Оцени несколько фильмов по 4 критериям – система найдёт похожего пользователя в базе и порекомендует фильмы на основе его предпочтений."
    )

    movie_titles = movies_df["movie_title"].tolist()
    movie_id_by_title = dict(zip(movies_df["movie_title"], movies_df["movie_id"]))

    st.subheader("Выбери фильмы которые ты смотрел")
    selected_titles = st.multiselect(
        "Фильмы",
        movie_titles,
        max_selections=10,
        placeholder="Начни вводить название...",
    )

    user_ratings = {}
    if selected_titles:
        st.subheader("Оцени каждый фильм по критериям (1–5)")

        for title in selected_titles:
            with st.expander(f"{title}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    plot = st.slider("Сюжет", 1.0, 5.0, 3.0, 0.5, key=f"plot_{title}")
                with c2:
                    visual = st.slider("Визуал", 1.0, 5.0, 3.0, 0.5, key=f"vis_{title}")
                with c3:
                    acting = st.slider("Актёры", 1.0, 5.0, 3.0, 0.5, key=f"act_{title}")
                with c4:
                    emotion = st.slider(
                        "Атмосфера", 1.0, 5.0, 3.0, 0.5, key=f"emo_{title}"
                    )

                overall = round((plot + visual + acting + emotion) / 4, 2)
                st.caption(f"Итоговая оценка (среднее): **{overall}**")
                user_ratings[movie_id_by_title[title]] = {
                    "overall": overall,
                    "plot": plot,
                    "visual": visual,
                    "acting": acting,
                    "emotion": emotion,
                }

    if user_ratings and st.button("Найти рекомендации", type="primary"):
        rated_ids = set(user_ratings.keys())

        with st.spinner("Ищем похожего пользователя..."):
            best_user_idx = None
            best_similarity = -1

            new_vec = np.array(
                [user_ratings[mid]["overall"] for mid in sorted(rated_ids)]
            )

            for uid, group in full_df.groupby("user_idx"):
                common = rated_ids & set(group["movie_id"].tolist())
                if len(common) < 1:
                    continue
                common_sorted = sorted(common)
                existing_vec = np.array(
                    [
                        group[group["movie_id"] == mid]["overall"].values[0]
                        for mid in common_sorted
                    ]
                )
                new_common = np.array(
                    [user_ratings[mid]["overall"] for mid in common_sorted]
                )

                norm = np.linalg.norm(existing_vec) * np.linalg.norm(new_common)
                if norm == 0:
                    continue
                sim = float(np.dot(existing_vec, new_common) / norm)
                if sim > best_similarity:
                    best_similarity = sim
                    best_user_idx = uid

        if best_user_idx is None:
            st.warning(
                "Не найдено похожих пользователей. Попробуй выбрать более популярные фильмы."
            )
        else:
            watched_ids = set(
                full_df[full_df["user_idx"] == best_user_idx]["movie_id"].tolist()
            )
            watched_ids |= rated_ids

            recs, avg_weights = recommend_for_user(
                model,
                best_user_idx,
                watched_ids,
                valid_item_idxs,
                valid_movie_ids,
                meta_tensor,
                top_n,
            )

            st.success(f"Найден похожий пользователь (сходство: {best_similarity:.3f})")

            new_criteria_means = {
                c: np.mean([user_ratings[mid][c] for mid in user_ratings])
                for c in ["plot", "visual", "acting", "emotion"]
            }
            new_overall_mean = np.mean(
                [user_ratings[mid]["overall"] for mid in user_ratings]
            )

            raw_weights = np.array(
                [
                    new_criteria_means["plot"] - new_overall_mean + 3.0,
                    new_criteria_means["visual"] - new_overall_mean + 3.0,
                    new_criteria_means["acting"] - new_overall_mean + 3.0,
                    new_criteria_means["emotion"] - new_overall_mean + 3.0,
                ]
            )
            raw_weights = np.exp(raw_weights) / np.exp(raw_weights).sum()
            avg_weights = raw_weights

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Твой профиль весов критериев")
                criteria_labels = ["Сюжет", "Визуал", "Актёры", "Атмосфера"]
                weights_df = pd.DataFrame(
                    {
                        "Критерий": criteria_labels,
                        "Вес": avg_weights,
                    }
                )
                st.bar_chart(weights_df.set_index("Критерий"))

                top_criterion = criteria_labels[int(np.argmax(avg_weights))]
                st.info(f"Для тебя важнее всего: **{top_criterion}**")

            with col2:
                st.subheader(f"Топ-{top_n} рекомендаций")
                recs_enriched = []
                for r in recs:
                    movie_row = movies_df[movies_df["movie_id"] == r["movie_id"]]
                    title = (
                        movie_row["movie_title"].values[0]
                        if not movie_row.empty
                        else str(r["movie_id"])
                    )
                    genre_str = (
                        str(movie_row["movie_genres"].values[0])
                        if not movie_row.empty
                        else ""
                    )
                    genres = []
                    for g in genre_str.split(","):
                        try:
                            genres.append(genre_names.get(int(g.strip()), ""))
                        except ValueError:
                            pass

                    recs_enriched.append(
                        {
                            "Фильм": title,
                            "Жанры": ", ".join(genres[:3]),
                            "Прогноз": round(r["score"], 2),
                            "Сюжет": round(r["plot"], 2),
                            "Визуал": round(r["visual"], 2),
                            "Актёры": round(r["acting"], 2),
                            "Атмосфера": round(r["emotion"], 2),
                        }
                    )

                recs_df = pd.DataFrame(recs_enriched)
                st.dataframe(
                    recs_df,
                    use_container_width=True,
                    column_config={
                        "Прогноз": st.column_config.ProgressColumn(
                            "Прогноз", min_value=1, max_value=5, format="%.2f"
                        ),
                    },
                )

st.divider()
st.caption(
    "Модель: Multi-Criteria Neural CF с динамическими персональными весами · PyTorch · Датасет: MovieLens 1M + Amazon Reviews (синтетические критерии)"
)
