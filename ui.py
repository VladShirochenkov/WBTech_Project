import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import math
import numpy as np

st.set_page_config(page_title="Анализ Взаимодействий Пользователей", layout="wide")

st.title("Анализ Взаимодействий Пользователей и Рекомендаций")

@st.cache_data
def load_data():
    items = pd.read_parquet('data/processed/321.parquet')
    first_train = pd.read_parquet('data/processed/first_train_2.parquet')
    listwise_df = pd.read_parquet('data/processed/listwise_df_text_final.parquet')
    warm_test = pd.read_parquet('data/processed/warm_test_2.parquet')
    recommendations_df = pd.read_parquet('data/processed/recommendations_test.parquet')
    return items, first_train, listwise_df, warm_test, recommendations_df

items, first_train, listwise_df, warm_test, recommendations_df = load_data()

first_train = first_train.sort_values('datetime')

interactions_count = first_train.groupby('user_id').size().reset_index(name='interactions')

users_in_first_train = set(first_train['user_id'].unique())
users_in_listwise = set(listwise_df['user_id'].unique())
common_users = users_in_first_train.intersection(users_in_listwise)

interactions_count = interactions_count[interactions_count['user_id'].isin(common_users)]

item_to_filename = items.reset_index().set_index('item_id')['index'].to_dict()

def display_images(item_ids, item_mapping, image_folder='images', highlight_items=None):
    num_images = len(item_ids)
    cols = 5
    rows = math.ceil(num_images / cols)
    fig, ax = plt.subplots(rows, cols, figsize=(15, 3 * rows))

    ax = ax.flatten() if isinstance(ax, np.ndarray) else [ax]
    
    for idx, item_id in enumerate(item_ids):
        ax_current = ax[idx]
        file_name = item_mapping.get(item_id, None)
        
        if file_name:
            image_path = os.path.join(image_folder, f"{file_name}.jpg")
            if os.path.exists(image_path):
                try:
                    img = plt.imread(image_path)
                    ax_current.imshow(img)
                    ax_current.axis('off')
                    ax_current.set_title(f"Item {item_id}", fontsize=8)
                    
                    if highlight_items and item_id in highlight_items:
                        rect = Rectangle((0, 0), 1, 1, transform=ax_current.transAxes, linewidth=5, edgecolor='green', facecolor='none')
                        ax_current.add_patch(rect)
                except Exception as e:
                    st.write(f"Ошибка при загрузке изображения {image_path}: {e}")
                    ax_current.text(0.5, 0.5, 'Ошибка загрузки', ha='center', va='center')
                    ax_current.axis('off')
            else:
                ax_current.text(0.5, 0.5, 'Изображение не найдено', ha='center', va='center')
                ax_current.axis('off')
        else:
            ax_current.text(0.5, 0.5, 'Неизвестный Item ID', ha='center', va='center')
            ax_current.axis('off')

    for j in range(num_images, rows * cols):
        ax[j].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)

st.sidebar.header("Фильтрация Пользователей по Количеству Взаимодействий")

quantile_ranges = [
    "0-50%", "51-60%", "61-70%", "71-80%", "81-90%", "91-100%"]

quantile_selection = st.sidebar.selectbox(
    "Выберите диапазон квантилей:",
    options=quantile_ranges
)

def parse_quantile_range(q_range):
    lower, upper = q_range.strip('%').split('-')
    lower = float(lower) / 100
    upper = float(upper) / 100
    return lower, upper

lower_q, upper_q = parse_quantile_range(quantile_selection)

lower_value = interactions_count['interactions'].quantile(lower_q)
upper_value = interactions_count['interactions'].quantile(upper_q)

filtered_users = interactions_count[
    (interactions_count['interactions'] >= lower_value) &
    (interactions_count['interactions'] < upper_value)
]

min_interactions = filtered_users['interactions'].min()
max_interactions = filtered_users['interactions'].max()

st.sidebar.markdown(f"**Диапазон квантилей:** {quantile_selection}")
st.sidebar.markdown(f"**Минимум взаимодействий в этом квантиле:** {min_interactions}")
st.sidebar.markdown(f"**Максимум взаимодействий в этом квантиле:** {max_interactions}")

if filtered_users.empty:
    st.warning("Нет пользователей, соответствующих выбранному квантилю.")
else:
    random_top_user = random.choice(filtered_users['user_id'].tolist())
    st.subheader(f"Анализ для пользователя: {random_top_user}")

    user_interactions = first_train[first_train['user_id'] == random_top_user]
    user_interactions_sorted = user_interactions.sort_values('datetime', ascending=False)
    last_5_interactions = user_interactions_sorted.head(5)['item_id'].tolist()

    user_recommendations = listwise_df[listwise_df['user_id'] == random_top_user].sort_values(by='rank').head(10)['item_id'].tolist()

    user_recommendations_df = recommendations_df[
        (recommendations_df['user_id'] == random_top_user) &
        (recommendations_df['item_id'].isin(user_recommendations))
    ]

    interacted_recommendations = user_recommendations_df[user_recommendations_df['weight'] == 1]['item_id'].tolist()

    st.markdown("### Последние взаимодействия")
    display_images(
        last_5_interactions,
        item_to_filename,
        image_folder='images'
    )

    st.markdown("### Первые 10 рекомендаций")
    display_images(
        user_recommendations,
        item_to_filename,
        image_folder='images',
        highlight_items=interacted_recommendations
    )

    user_real_interactions = warm_test[warm_test['user_id'] == random_top_user]['item_id'].tolist()

    st.markdown("### Реальные взаимодействия (warm_test)")
    display_images(
        user_real_interactions,
        item_to_filename,
        image_folder='images'
    )
