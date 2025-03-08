# Streamlit e plugins
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report
from streamlit_folium import st_folium
from collections import Counter

# Análise e manipulação de dados
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from ydata_profiling import ProfileReport

# Machine Learning (Sklearn)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve, auc)
from sklearn.preprocessing import (PolynomialFeatures, MinMaxScaler, OneHotEncoder,
                                   StandardScaler, label_binarize)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA

# Imbalance Learning (Imblearn)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Visualização geoespacial
import folium
from folium.plugins import HeatMap

# Nuvem de palavras
from wordcloud import WordCloud




def main():
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
            options=["Home", "Filtragem", "Mapa", "Profiling", "Classificação", "Clusterização"], 
            icons=['bi-house', 'bi-search', 'bi-geo-alt-fill', 'bi-file-earmark-zip-fill', 'bi-hand-thumbs-up', 'bi-diagram-3'], 
            menu_icon="laptop", 
            default_index=0,  
        )





    # TELA DE FILTRAGEM DE DADOS
    if selecionado == 'Filtragem':

        opcao_selecionada = option_menu(
        menu_title='DATAFRAMES', 
        options=["Explorar", "Filtrar"],  
        icons=['journal-text', 'list-check'],  
        menu_icon="map",  
        default_index=0,  
        orientation="horizontal"  
    )
        if opcao_selecionada == 'Explorar':

            # Cria duas colunas para organizar os selectboxes lado a lado
            col1, col2 = st.columns(2)

            # Selectbox para escolher o dataset
            with col1:
                dataset_selecionado = st.selectbox(
                    "Selecione um dataset:",
                    ['df_negocios', 'df_tip']
                )

            # Define o DataFrame correspondente
            df = df_negocios if dataset_selecionado == 'df_negocios' else df_tip

            # Selectbox para escolher a coluna de ordenação (somente após selecionar o dataset)
            with col2:
                coluna_ordenacao = st.selectbox(
                    "Selecione a coluna para ordenar:",
                    df.columns.tolist()
                )

            # Ordena o DataFrame pela coluna selecionada
            df_ordenado = df.sort_values(by=coluna_ordenacao)

            # Exibe o DataFrame ordenado (clicável para reordenar)
            st.dataframe(df_ordenado)
        


        elif opcao_selecionada == 'Filtrar':
            with st.expander("Tendência de avaliações ao longo do tempo"):
                df_tip['date'] = pd.to_datetime(df_tip['date'], errors='coerce')

                st.title('Análise de Tendência de Avaliações')

                # Criar colunas para mês e ano
                col1, col2 = st.columns(2)

                # Seleção do ano e mês
                with col1:
                    ano = st.selectbox("Selecione o ano:", sorted(df_tip['date'].dt.year.unique(), reverse=True))
                with col2:
                    mes = st.selectbox("Selecione o mês:", sorted(df_tip['date'].dt.month.unique()))

                # Filtrar df_tip pelo período selecionado
                df_tip_filtrado = df_tip[(df_tip['date'].dt.year == ano) & (df_tip['date'].dt.month == mes)]

                # Unir df_tip com df_negocios para obter as estrelas dos estabelecimentos
                df_merged = df_tip_filtrado.merge(df_negocios[['business_id', 'stars']], on='business_id', how='left')

                # Agrupar por dia e calcular a média das estrelas
                df_tendencia = df_merged.groupby(df_merged['date'].dt.day)['stars'].mean().reset_index()
                df_tendencia.rename(columns={'date': 'day'}, inplace=True)

                # Verificar se há dados para exibir
                if df_tendencia.empty:
                    st.warning("Nenhuma avaliação encontrada para o período selecionado.")
                else:
                    # Criar gráfico de tendência usando Plotly
                    fig = px.line(
                        df_tendencia, x='day', y='stars', markers=True,
                        title=f'Tendência de Avaliações ({mes}/{ano})', labels={'day': 'Dia do Mês', 'stars': 'Média de Estrelas'}
                    )
                    
                    # Mostrar gráfico no Streamlit
                    st.plotly_chart(fig)
                    
                    # GRÁFICO TENDENCIAS 2009 - 2022
                    df_tip['date'] = pd.to_datetime(df_tip['date'], errors='coerce')

                    df_tip['year_month'] = df_tip['date'].dt.to_period('M').astype(str)
                    stars_over_time = df_tip.merge(df_negocios[['business_id', 'stars']], on='business_id') \
                                            .groupby('year_month')['stars'].mean().reset_index()

                    fig3 = px.line(
                        stars_over_time, x='year_month', y='stars',
                        title='Tendências de Avaliações (2009 - 2022)', labels={'year_month': 'Mês_Ano', 'stars': 'Estrelas'}
                    )
                    
                    st.plotly_chart(fig3)

            with st.expander("Estabelecimentos por faixa de preço"):
                fig2 = px.histogram(
                    df_negocios, x='RestaurantsPriceRange2',
                    nbins=4, color_discrete_sequence=['skyblue'],
                    title='Faixa de Preço',
                    labels={'RestaurantsPriceRange2': 'Faixa de Preço'}
                )
                st.plotly_chart(fig2)
                
                st.subheader("Filtrar Estabelecimentos/Categorias por Faixa de Preço")

                # Criar colunas para o selectbox e slider
                col1, col2 = st.columns(2)

                # Selectbox para escolher a faixa de preço
                with col1:
                    faixa_preco = st.selectbox(
                        "Selecione a faixa de preço:",
                        [1, 2, 3, 4]
                    )

                # Slider para escolher o número de categorias a serem mostradas
                with col2:
                    numero_de_categorias = st.slider(
                        "Selecione o número de categorias a exibir:",
                        min_value=5, max_value=50, value=5, step=5
                    )

                # Filtra os dados de acordo com a faixa de preço selecionada
                df_filtrado = df_negocios[df_negocios['RestaurantsPriceRange2'] == faixa_preco]

                # Exibe o DataFrame filtrado
                col1_df, col2_df = st.columns(2)  # Criar colunas para os dois dataframes

                with col1_df:
                    st.write(f"Quantidade de Estabelecimentos: {df_filtrado.shape[0]}")
                    st.dataframe(df_filtrado)  # Exibe o primeiro dataframe com os estabelecimentos filtrados
                
                # Exibir a tabela com as categorias mais citadas
                todas_as_categorias = []

                # Iterando sobre a coluna 'categories' do df_filtrado
                for categorias in df_filtrado['categories'].dropna():
                    todas_as_categorias.extend(categorias.split(', '))

                # Contar a frequência das categorias
                categoria_contagem = Counter(todas_as_categorias)

                # Obter as categorias mais comuns com base no valor do slider
                categorias_mais_comuns = categoria_contagem.most_common(numero_de_categorias)

                # Criar um DataFrame para exibir como tabela
                categorias_df = pd.DataFrame(categorias_mais_comuns, columns=['Categoria', 'Quantidade'])

                with col2_df:
                    st.write("Categorias mais comuns por faixa de preço")
                    st.dataframe(categorias_df)  # Exibe o segundo dataframe com as categorias mais comuns


    
    # TELA ANALISE EXPLORATORIA

    # Exibe o conteúdo com base na opção selecionada
    elif selecionado == 'Profiling':
        
        st.write('''
                 # ANALISE EXPLORATÓRIA''')

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


        opcao_selecionada = option_menu(
        menu_title='PÁGINA INICIAL', 
        options=["Resumo", "Objetivos", "Perguntas","Integrantes"],  
        icons=['journal-text', 'list-check', 'bi-question-circle-fill', 'bi-person-fill-add'],  
        menu_icon="cast",  
        default_index=0,  
        orientation="horizontal"  
    )
        
        if opcao_selecionada == 'Resumo':
            st.write('''### RESUMO
O turismo regional apresenta-se como uma alternativa sustentável para descentralizar o fluxo turístico de grandes centros urbanos, promovendo a valorização de destinos menos explorados. Este trabalho propõe o desenvolvimento de um aplicativo voltado ao turismo regional, integrando funcionalidades de planejamento de itinerários e avaliações de estabelecimentos. Combinando técnicas de aprendizado de máquina, como clusterização e classificação, o presente estudo busca identificar padrões de popularidade e categorias de negócios em destinos regionais, além de prever faixas de preço de estabelecimentos com base em dados do Yelp. Entre as funcionalidades, destacam-se a criação de itinerários organizados, gestão de tarefas de viagem e avaliações colaborativas, além da integração com serviços como APIs de geolocalização. A validação da proposta será conduzida por meio de uma análise baseada em ciência de dados, explorando padrões em avaliações e correlacionando características de negócios com suas faixas de preço.
''')
        elif opcao_selecionada == 'Objetivos':
            st.write('''### OBJETIVO DO PROJETO
Desenvolver uma solução integrada voltada ao turismo regional, composta por um aplicativo para planejamento de itinerários, avaliações de pontos turísticos e estabelecimentos, e algoritmos de aprendizado de máquina. Essa solução visa promover destinos menos explorados, personalizar experiências com base nos interesses dos usuários e oferecer insights estratégicos para a gestão e planejamento do turismo em áreas regionais.

### Objetivos Específicos:
- **Desenvolver um aplicativo funcional para dispositivos móveis:**
    - Permitir aos usuários explorar destinos regionais.
    - Integrar um sistema de avaliações e recomendações para facilitar o planejamento de viagens.
- **Realizar uma análise exploratória de dados:**
    - Utilizar dados do Yelp e outras fontes para identificar padrões nas avaliações e características de estabelecimentos.
    - Correlacionar fatores como faixa de preço, popularidade e localização com as preferências dos usuários.
- **Implementar algoritmos de aprendizado de máquina:**
    - Aplicar técnicas de clusterização para categorizar estabelecimentos e pontos turísticos com base em características comuns.
    - Utilizar classificação para prever tendências de comportamento dos usuários e preferências no turismo regional.''')
        elif opcao_selecionada == 'Perguntas':
            st.write('''### PERGUNTAS DE PESQUISA
- Quais padrões de características de negócios podem ser identificados por meio de métodos de aprendizado não supervisionado, como clusterização, para entender a popularidade e as categorias de estabelecimentos em destinos turísticos regionais?

- Quais algoritmos de aprendizado supervisionado oferecem maior precisão na predição de faixas de preço de estabelecimentos regionais com base em dados do Yelp?

- Como a interação entre turistas e estabelecimentos locais(avaliações)pode ser analisada para otimizar a recomendação de destinos e atividades com base no perfil do usuário e nas avaliações compartilhadas por outros turistas? 
''')
        elif opcao_selecionada == 'Integrantes':
            st.write('''### INTEGRANTES DO PROJETO
- Ellen Caroliny
- Evny Vitória
- Igor Queiroz
- Isadora Albuquerque
- Stéphanie Cândido''')


    
    
    
    # TELA DE CLUSTERIZAÇÃO
    
    elif selecionado == 'Clusterização':
        #df = pd.read_csv('./dataset/yelp_academic_dataset_business_cleaned.csv')

        st.title("Agrupamento com a base de dados yelp_academic_dataset")

        original_df = pd.read_parquet('./dataset/yelp_academic_dataset_business.parquet')

        columns = ['stars', 'review_count', 'latitude', 'longitude', 'categories']

        original_df = original_df[columns]
        with st.expander('Dataframe Original'):
            st.write("Dataframe yelp_academic_dataset_business com as colunas selecionadas para clusterização")
            st.subheader("Dados")
            st.dataframe(original_df.head(10000))
            st.subheader("Descrição")
            st.dataframe(original_df.head(10000).describe())

        # Reduzir o número de amostras para 10.000
        sampled_df = df_negocios.sample(n=10000, random_state=2)

        # Selecionar as colunas relevantes para a clusterização
        #columns = ['stars', 'review_count', 'latitude', 'longitude', 'categories']
        sampled_df = sampled_df[columns]

        with st.expander("Dataframe Modificado"):
            st.write("Dataframe com as cinco colunas selecionadas após embaralhamento e remoção dos valores nulos")
            st.subheader("Dados")
            st.dataframe(sampled_df)
            st.subheader("Descrição")
            st.dataframe(sampled_df.describe())

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

        # CLUSTERIZAÇÃO - Método do Cotovelo
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
        with st.expander('Cotovelo'):
            st.write('com as 5 colunas' )
            st.pyplot(plt)

        # CLUSTERIZAÇÃO COM KMEANS
        with st.expander("Clusterização com KMeans"):
            st.subheader("Selecione as colunas para clusterização")
            selected_columns = st.multiselect(
                "Escolha pelo menos 2 colunas:",
                numeric_features + categorical_features,  # Adiciona as colunas numéricas e categóricas
                default=numeric_features  # Seleciona as colunas numéricas por padrão
            )

            # Verifica se o usuário selecionou pelo menos duas colunas
            if len(selected_columns) >= 2:
                st.subheader("Selecione o número de clusters")
                qtd_clusters = st.slider(
                    "Escolha o número de clusters (entre 2 e 10):",
                    min_value=2,
                    max_value=10,
                    value=6  # Valor padrão
                )

                # Ajuste a transformação com as colunas selecionadas
                numeric_selected = [col for col in selected_columns if col in numeric_features]
                categorical_selected = [col for col in selected_columns if col in categorical_features]

                # Transformador para dados numéricos (normalização)
                numeric_transformer = MinMaxScaler()

                # Transformador para dados categóricos (One-Hot Encoding)
                categorical_transformer = OneHotEncoder(handle_unknown='ignore')

                # Criar um pipeline de transformação
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_selected),
                        ('cat', categorical_transformer, categorical_selected)
                    ])

                # Aplicar a transformação
                processed_data = preprocessor.fit_transform(sampled_df)

                # Aplicar KMeans
                kmeans_model = KMeans(n_clusters=qtd_clusters, random_state=2)
                kmeans_result = kmeans_model.fit_predict(processed_data)

                # Reduzir a dimensionalidade para 2D com PCA para o gráfico de dispersão
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(processed_data)

                # Plotar os clusters
                plt.figure(figsize=(8, 6))
                plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_result, cmap='viridis', s=50, alpha=0.5)

                # Adicionar títulos e rótulos aos eixos
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                plt.title('KMeans Clustering (Dados Selecionados)')

                # Exibir o gráfico
                st.pyplot(plt)
            else:
                st.warning("Por favor, selecione pelo menos duas colunas para realizar a clusterização.")












    # CLASSIFICAÇÃO
    elif selecionado == 'Classificação':
        #Logistic Regression
        # Carregar o dataset
        df = pd.read_parquet('../dataset/yelp_academic_dataset_business_cleaned.parquet')

        # Seleção de colunas
        cols = [
            'stars', 'review_count', 'latitude', 'longitude',
            'Alcohol', 'BikeParking', 'RestaurantsDelivery', 'RestaurantsTakeOut', 'RestaurantsPriceRange2'
        ]
        df = df[cols]

        # Converter colunas categóricas
        df['Alcohol'] = df['Alcohol'].map({'None': 0, 'Beer&Wine': 1, 'Full_Bar': 2, 'Unknown': 1})
        df['BikeParking'] = df['BikeParking'].fillna(0).astype(int)
        df['RestaurantsDelivery'] = df['RestaurantsDelivery'].map({'True': 1, 'False': 0, 'Unknown': 0})
        df['RestaurantsTakeOut'] = df['RestaurantsTakeOut'].map({'True': 1, 'False': 0, 'Unknown': 0})

        # Remover valores nulos na coluna alvo
        df = df.dropna(subset=['RestaurantsPriceRange2'])

        # Combinar classes 3 e 4
        df['RestaurantsPriceRange2'] = df['RestaurantsPriceRange2'].replace({4: 3}) - 1

        # Separar features e target
        X = df.drop('RestaurantsPriceRange2', axis=1)
        y = df['RestaurantsPriceRange2'].astype(int)

        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

        # Pré-processamento
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), ['stars', 'review_count', 'latitude', 'longitude']),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent'))
                ]), ['Alcohol', 'BikeParking', 'RestaurantsDelivery', 'RestaurantsTakeOut'])
            ]
        )

        # Pipeline com SMOTEENN e Regressão Logística
        pipeline_lr_smoteenn = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smoteenn', SMOTEENN(random_state=42)),  # SMOTEENN aqui
            ('classifier', OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=1000, random_state=1)))
        ])

        # Treinar o modelo
        pipeline_lr_smoteenn.fit(X_train, y_train)

        # Previsões no conjunto de teste
        y_pred_lr_smoteenn = pipeline_lr_smoteenn.predict(X_test)

        # Matriz de Confusão e Relatório de Classificação
        cm_lr_smoteenn = confusion_matrix(y_test, y_pred_lr_smoteenn)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_lr_smoteenn, annot=True, fmt='d', cmap='Blues', xticklabels=['Baixo', 'Médio', 'Alto'], yticklabels=['Baixo', 'Médio', 'Alto'])
        plt.xlabel('Previsão')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão - Regressão Logística (SMOTEENN)')
        plt.show()

if __name__ == '__main__':
    main()