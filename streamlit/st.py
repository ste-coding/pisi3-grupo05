import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from wordcloud import WordCloud
from imblearn.pipeline import Pipeline as ImbPipeline

# Carregando os datasets necessários
@st.cache_data
def load_data():
    df_negocios = pd.read_parquet('./dataset/yelp_academic_dataset_business_cleaned.parquet')
    df_tip = pd.read_parquet('./dataset/yelp_academic_dataset_tip_cleaned.parquet')
    return df_negocios, df_tip

df_negocios, df_tip = load_data()

#Ler arquivos de texto
def ler_arquivo(nome_arquivo):
    with open(nome_arquivo, 'r', encoding='utf-8') as file:
        return file.read()


# Menu horizontal usando option_menu
with st.sidebar:
    selecionado = option_menu(
        menu_title="PISI3",  # Título do menu
        options=["Home", "Profiling", "Mapa", "Classificação", "Clusterização"], 
        icons=['bi-house', 'bi-file-earmark-zip-fill', 'bi-geo-alt-fill', 'bi-hand-thumbs-up', 'bi-diagram-3'], 
        menu_icon="laptop", 
        default_index=0,  
    )


# TELA ANALISE EXPLORATORIA

# Exibe o conteúdo com base na opção selecionada
if selecionado == 'Profiling':
    analise = 'textos/analise.txt'
    st.write(ler_arquivo(analise))

    opcao_selecionada = option_menu(
    menu_title=None, 
    options=["Profiling", "Análise Exploratória"],  
    icons=['journal-text','bi-search'],  
    menu_icon="cast",  
    default_index=0,  
    orientation="horizontal"  
)
    # SELECIONAR DATASET PARA YDATA PROFILING
    if opcao_selecionada == 'Profiling':
        dataset_selecionado = st.selectbox(
        "Selecione um dataset:",
        ["df_negocios", "df_tip"]
    )   
        
        if st.button("Analisar"):
            # DF_TIP
            if dataset_selecionado == "df_tip":
                st.header("Profiling do Dataset Tip")
                profile = ProfileReport(df_tip, explorative=True)
                st_profile_report(profile)

            # DF_BUSINESS
            elif dataset_selecionado == "df_negocios":
                st.header("Profiling do Dataset Negócios")
                profile = ProfileReport(df_negocios, explorative=True)
                st_profile_report(profile)
    else:
        # TOP 10 LUGARES MAIS BEM AVALIADOS
        top_lugares = df_negocios.sort_values(by='stars', ascending=False).head(10)
        fig1 = px.bar(
        top_lugares,
        x='name', y='stars',
        title='Top 10 lugares mais bem avaliados',
        labels={'name': 'Lugar', 'stars': 'Avaliação'},
        color='stars'
    )
        with st.expander("Top 10 lugares mais bem avaliados"):
            st.plotly_chart(fig1)

    # DISTRIBUIÇÃO FAIXA DE PREÇO        
        fig2 = px.histogram(
        df_negocios, x='RestaurantsPriceRange2',
        nbins=4, color_discrete_sequence=['skyblue'],
        title='Faixa de Preço',
        labels={'RestaurantsPriceRange2': 'Faixa de Preço'}
    )
        with st.expander("Distribuição da Faixa de Preço"):
            st.plotly_chart(fig2)
        
    # TENDÊNCIAS TEMPORAIS
        df_tip['date'] = pd.to_datetime(df_tip['date'], errors='coerce')

        df_tip['year_month'] = df_tip['date'].dt.to_period('M').astype(str)
        stars_over_time = df_tip.merge(df_negocios[['business_id', 'stars']], on='business_id') \
                                .groupby('year_month')['stars'].mean().reset_index()

        fig3 = px.line(
            stars_over_time,
            x='year_month', y='stars', title='Tendências das avaliações ao longo do tempo'
        )
        with st.expander("Tendência das avaliações ao longo do tempo"):
            st.plotly_chart(fig3)

    # TOP 10 CIDADES COM ESTABELECIMENTOS MAIS BEM AVALIADOS
        estabelecimentos_bem_avaliados = df_negocios[df_negocios['stars'] >= 4]

        cidade_bem_avaliada = estabelecimentos_bem_avaliados.groupby('city').agg(
            media_avaliacao=('stars', 'mean'),
            total_estabelecimentos=('business_id', 'count')
        ).sort_values(by='total_estabelecimentos', ascending=False)

        # Gráfico de barras
        fig4, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(x=cidade_bem_avaliada['total_estabelecimentos'].head(10), y=cidade_bem_avaliada.head(10).index, ax=ax)
        ax.set_title('Top 10 Cidades com Mais Estabelecimentos Bem Avaliados')
        ax.set_xlabel('Número de Estabelecimentos Bem Avaliados')
        ax.set_ylabel('Cidades')

        with st.expander("Cidades mais bem avaliadas"):
            st.pyplot(fig4)

    # NUVEM DE COMENTÁRIOS
        text = " ".join(df_tip['text'].dropna())
        wordcloud = WordCloud(width=800, height=400).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        with st.expander("Análise Textual"):
            st.pyplot(plt)
            st.write("Palavras mais encontradas nos comentários")

# TELA DO MAPA
elif selecionado == "Mapa":
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

# TELA INICIAL
elif selecionado == 'Home':
    main = 'textos/main.txt'
    objetivos = 'textos/objetivos.txt'
    pps = 'textos/pps.txt'
    integrantes = 'textos/integrantes.txt'

    st.title("PISI 3 - PÁGINA INICIAL")
    opcao_selecionada = option_menu(
    menu_title=None, 
    options=["Resumo", "Objetivos", "Perguntas","Integrantes"],  
    icons=['journal-text', 'list-check', 'bi-question-circle-fill', 'bi-person-fill-add'],  
    menu_icon="cast",  
    default_index=0,  
    orientation="horizontal"  
)
    
    if opcao_selecionada == 'Resumo':
        st.write(ler_arquivo(main))
    elif opcao_selecionada == 'Objetivos':
        st.write(ler_arquivo(objetivos))
    elif opcao_selecionada == 'Perguntas':
        st.write(ler_arquivo(pps))
    elif opcao_selecionada == 'Integrantes':
        st.write(ler_arquivo(integrantes))


elif selecionado == 'Clusterização':
    df = pd.read_csv('./dataset/yelp_academic_dataset_business_cleaned.csv')
    # Reduzir o número de amostras para 10.000
    sampled_df = df.sample(n=10000, random_state=2)

    # Selecionar as colunas relevantes para a clusterização
    columns = ['stars', 'review_count', 'latitude', 'longitude', 'categories']
    sampled_df = sampled_df[columns]

    # Tratamento de dados categóricos e numéricos
    numeric_features = ['stars', 'review_count', 'latitude', 'longitude']
    categorical_features = ['categories']

    # Transformador para dados numéricos (normalização)
    numeric_transformer = MinMaxScaler()

    # Transformador para dados categóricos (One-Hot Encoding)
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Criar um pipeline de transformação
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Aplicar a transformação
    processed_data = preprocessor.fit_transform(sampled_df)

    # clusterização - Método do Cotovelo
    inertia = []
    for n in range(1, 11):
        kmeans = KMeans(n_clusters=n, random_state=2)
        kmeans.fit(processed_data)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo')
    st.pyplot(plt)



# CLASSIFICAÇÃO
elif selecionado == 'Classificação':
    #Logistic Regression
    df = pd.read_csv('./dataset/yelp_academic_dataset_business_cleaned.csv')

    cols = [
        'stars', 'review_count', 'latitude', 'longitude',
        'Alcohol', 'BikeParking', 'RestaurantsDelivery', 
        'RestaurantsTakeOut', 'RestaurantsPriceRange2'
    ]
    df = df[cols]

    # Converter colunas categóricas
    df['Alcohol'] = df['Alcohol'].map({'None': 0, 'Beer&Wine': 1, 'Full_Bar': 2, 'Unknown': 1})
    df['BikeParking'] = df['BikeParking'].fillna(0).astype(int)
    df['RestaurantsDelivery'] = df['RestaurantsDelivery'].map({'True': 1, 'False': 0, 'Unknown': 0})
    df['RestaurantsTakeOut'] = df['RestaurantsTakeOut'].map({'True': 1, 'False': 0, 'Unknown': 0})

    df = df.dropna(subset=['RestaurantsPriceRange2'])

    X = df.drop('RestaurantsPriceRange2', axis=1)
    y = df['RestaurantsPriceRange2'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), ['stars', 'review_count', 'latitude', 'longitude']),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('passthrough', 'passthrough')
            ]), ['Alcohol', 'BikeParking', 'RestaurantsDelivery', 'RestaurantsTakeOut'])
        ]
    )

    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=1)),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=1))
    ])
    y_train_bin = label_binarize(y_train, classes=[1, 2, 3, 4])
    y_test_bin = label_binarize(y_test, classes=[1, 2, 3, 4])

    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=1000, random_state=1)))
    ])

    pipeline.fit(X_train, y_train_bin)

    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)

    cm = confusion_matrix(y_test, y_pred.argmax(axis=1) + 1)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Logistic Regression')
    st.pyplot(plt)
