import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from wordcloud import WordCloud

# Carregando os datasets necessários
@st.cache_data
def load_data():
    df_negocios = pd.read_parquet('yelp_academic_dataset_business_cleaned.parquet')
    df_tip = pd.read_parquet('yelp_academic_dataset_tip_cleaned.parquet')
    return df_negocios, df_tip

df_negocios, df_tip = load_data()

# Título principal do aplicativo Streamlit
st.title('Exploração Turística e Análise de Negócios do Yelp')

# Organização em tabelas
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Top Avaliações", "Faixa de Preço", "Mapas Interativos", "Tendências Temporais", "Análise Textual", "Cidades Bem Avaliadas", "Relacionamento de Comentários"
])

# Seção 1: Top Avaliações
with tab1:
    st.subheader('Top 10 lugares mais bem avaliados')
    top_lugares = df_negocios.sort_values(by='stars', ascending=False).head(10)
    fig1 = px.bar(
        top_lugares,
        x='name', y='stars',
        title='Top 10 lugares mais bem avaliados',
        labels={'name': 'Lugar', 'stars': 'Avaliação'},
        color='stars'
    )
    st.plotly_chart(fig1)

# Seção 2: Distribuição da Faixa de Preço
with tab2:
    st.subheader('Distribuição da Faixa de Preço')
    fig2 = px.histogram(
        df_negocios, x='RestaurantsPriceRange2',
        nbins=4, color_discrete_sequence=['skyblue'],
        title='Faixa de Preço',
        labels={'RestaurantsPriceRange2': 'Faixa de Preço'}
    )
    st.plotly_chart(fig2)

# Seção 3: Mapas Interativos
with tab3:
    st.subheader('Mapa de Calor: Lugares Populares')
    mapa = folium.Map(location=[df_negocios['latitude'].mean(), df_negocios['longitude'].mean()], zoom_start=10)
    heat_data = [[row['latitude'], row['longitude'], row['stars']] for index, row in df_negocios.iterrows()]
    HeatMap(heat_data).add_to(mapa)
    st_folium(mapa, width=700, height=500)

    st.subheader('Mapa de Estabelecimentos Bem Avaliados por Cidade')
    cidade_selecionada = st.selectbox(
        'Selecione uma cidade para explorar os 10 melhores estabelecimentos:',
        df_negocios['city'].unique()
    )

    estabelecimentos_na_cidade = df_negocios[df_negocios['city'] == cidade_selecionada]
    top_estabelecimentos = estabelecimentos_na_cidade.sort_values(by='stars', ascending=False).head(10)

    mapa_cidade = folium.Map(
        location=[top_estabelecimentos['latitude'].mean(), top_estabelecimentos['longitude'].mean()],
        zoom_start=12
    )

    for _, row in top_estabelecimentos.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{row['name']} ({row['stars']}⭐)",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(mapa_cidade)

    st_folium(mapa_cidade, width=700, height=500)

# Seção 4: Tendências Temporais
with tab4:
    st.subheader('Tendência temporal das avaliações')
    df_tip['date'] = pd.to_datetime(df_tip['date'], errors='coerce')

    df_tip['year_month'] = df_tip['date'].dt.to_period('M').astype(str)
    stars_over_time = df_tip.merge(df_negocios[['business_id', 'stars']], on='business_id') \
                            .groupby('year_month')['stars'].mean().reset_index()

    fig4 = px.line(
        stars_over_time,
        x='year_month', y='stars', title='Tendências das avaliações ao longo do tempo'
    )
    st.plotly_chart(fig4)


# Seção 5: Análise Textual
with tab5:
    st.subheader('Nuvem de Palavras dos Comentários')
    text = " ".join(df_tip['text'].dropna())
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Seção 6: Cidades Bem Avaliadas
with tab6:
    st.subheader('Top 10 Cidades com Mais Estabelecimentos Bem Avaliados')

    estabelecimentos_bem_avaliados = df_negocios[df_negocios['stars'] >= 4]

    cidade_bem_avaliada = estabelecimentos_bem_avaliados.groupby('city').agg(
        media_avaliacao=('stars', 'mean'),
        total_estabelecimentos=('business_id', 'count')
    ).sort_values(by='total_estabelecimentos', ascending=False)

    # Gráfico de barras
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(x=cidade_bem_avaliada['total_estabelecimentos'].head(10), y=cidade_bem_avaliada.head(10).index, ax=ax)
    ax.set_title('Top 10 Cidades com Mais Estabelecimentos Bem Avaliados')
    ax.set_xlabel('Número de Estabelecimentos Bem Avaliados')
    ax.set_ylabel('Cidades')
    st.pyplot(fig)

    st.write('Essas cidades destacam-se por terem um grande número de estabelecimentos bem avaliados.')

# Seção 7: Relacionamento entre Comentários e Avaliações
with tab7:
    st.subheader('Quantidade de Comentários x Avaliações')

    total_comments = df_tip.groupby('business_id').size().reset_index(name='total_comments')
    df_merged = df_negocios.merge(total_comments, on='business_id', how='left')

    # Jitter para dispersão
    jitter_strength = 0.1
    df_merged['stars_jittered'] = df_merged['stars'] + np.random.uniform(-jitter_strength, jitter_strength, size=df_merged.shape[0])
    df_merged['total_comments_jittered'] = df_merged['total_comments'] + np.random.uniform(-jitter_strength, jitter_strength, size=df_merged.shape[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_merged['stars_jittered'], df_merged['total_comments_jittered'], alpha=0.5, color='blue')
    ax.set_title('Quantidade de Comentários x Avaliações')
    ax.set_xlabel('Avaliações')
    ax.set_ylabel('Total de Comentários')
    ax.grid(False)
    st.pyplot(fig)
