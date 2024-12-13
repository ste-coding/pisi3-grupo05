import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

df_negocios = pd.read_parquet('yelp_academic_dataset_business_cleaned.parquet')
df_tip = pd.read_parquet('yelp_academic_dataset_tip_cleaned.parquet')

st.title('Análise de Negócios do Yelp')

category_counts = df_negocios['categories'].value_counts().head(10)

st.subheader('Top 10 Categorias Mais Frequentes')
fig, ax = plt.subplots(figsize=(14, 8))
sns.barplot(x=category_counts.values, y=category_counts.index, ax=ax)
ax.set_title('Top 10 Categorias Mais Frequentes')
ax.set_xlabel('Contagem')
ax.set_ylabel('Categorias')
st.pyplot(fig)

st.subheader('Distribuição da Faixa de Preço')
fig, ax = plt.subplots(figsize=(10, 6))
bins = [0.5, 1.5, 2.5, 3.5, 4.5]
sns.histplot(df_negocios['RestaurantsPriceRange2'], bins=bins, kde=False, color='skyblue', edgecolor='black', stat='frequency', ax=ax)
ax.set_xticks([1, 2, 3, 4])
ax.set_title('Distribuição da Faixa de Preço', fontsize=14)
ax.set_xlabel('Faixa de Preço', fontsize=12)
ax.set_ylabel('Frequência', fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), 
                textcoords='offset points')
st.pyplot(fig)

st.subheader('Distribuição das Estrelas')
fig, ax = plt.subplots(figsize=(10, 6))
hist_data = sns.histplot(df_negocios['stars'], bins=10, kde=False, ax=ax, color='skyblue', edgecolor='black')
hist_data.set_ylabel('Frequência')
x = [patch.get_x() + patch.get_width() / 2 for patch in hist_data.patches]
y = [patch.get_height() for patch in hist_data.patches]
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(np.array(x).reshape(-1, 1))
poly_reg_model = LinearRegression()
poly_reg_model.fit(x_poly, y)
x_full = np.linspace(min(x), max(x), 200)
x_full_poly = poly.fit_transform(x_full.reshape(-1, 1))
y_smooth = poly_reg_model.predict(x_full_poly)
ax.plot(x_full, y_smooth, color='red', linewidth=2, label='Curva Suave')
ax.set_title('Distribuição das Estrelas')
ax.set_xlabel('Estrelas')
ax.set_ylabel('Frequência')
ax.legend()
st.pyplot(fig)

st.subheader('Relação entre Faixa de Preço e Número de Estrelas ')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='RestaurantsPriceRange2', y='stars', data=df_negocios, ax=ax)
ax.set_title('Relação entre Faixa de Preço e Número de Estrelas ')
ax.set_xlabel('Faixa de Preço')
ax.set_ylabel('Estrelas')
st.pyplot(fig)

st.subheader('Relação entre Faixa de Preço e Número de Estrelas (Sem Outliers)')
Q1 = df_negocios['stars'].quantile(0.25)
Q3 = df_negocios['stars'].quantile(0.75)
IQR = Q3 - Q1
df_negocios_filtered = df_negocios[~((df_negocios['stars'] < (Q1 - 1.5 * IQR)) | (df_negocios['stars'] > (Q3 + 1.5 * IQR)))]
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='RestaurantsPriceRange2', y='stars', data=df_negocios_filtered, ax=ax)
ax.set_title('Relação entre Faixa de Preço e Número de Estrelas (Sem Outliers)')
ax.set_xlabel('Faixa de Preço')
ax.set_ylabel('Estrelas')
st.pyplot(fig)

st.subheader('Tendência da Média de Avaliações ao Longo do Tempo')
df_tip['year_month'] = df_tip['date'].dt.to_period('M')
stars_over_time = df_tip.merge(df_negocios[['business_id', 'stars']], on='business_id').groupby('year_month')['stars'].mean()
fig, ax = plt.subplots(figsize=(14, 8))
stars_over_time.plot(ax=ax)
ax.set_title('Tendência da Média de Avaliações ao Longo do Tempo')
ax.set_xlabel('Ano e Mês')
ax.set_ylabel('Média de Estrelas')
st.pyplot(fig)

st.subheader('Popularidade Regional: Mapa de Calor')
mapa = folium.Map(location=[df_negocios['latitude'].mean(), df_negocios['longitude'].mean()], zoom_start=10)
heat_data = [[row['latitude'], row['longitude'], row['stars']] for index, row in df_negocios.iterrows()]
HeatMap(heat_data).add_to(mapa)
st_folium(mapa, width=700, height=500)
