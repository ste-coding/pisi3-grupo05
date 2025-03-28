# Streamlit e plugins
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report
from streamlit_folium import st_folium
from collections import Counter
import joblib
from sklearn.metrics import silhouette_score, silhouette_samples

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
from imblearn.combine import SMOTEENN

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
        df_negocios = pd.read_parquet('../dataset/yelp_academic_dataset_business_cleaned.parquet')
        df_tip = pd.read_parquet('../dataset/yelp_academic_dataset_tip_cleaned.parquet')
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
        import matplotlib.cm as cm
        from sklearn.decomposition import PCA

        # Título da aplicação
        st.title("Clusterização de Dados com K-Means")

        # Carregar os dados diretamente do arquivo CSV
        @st.cache_data
        def load_data():
            df = pd.read_csv('../dataset/yelp_academic_dataset_business_cleaned.csv')
            return df

        df = load_data()

        # Reduzir para 10.000 amostras para melhor desempenho
        sampled_df = df.sample(n=10000, random_state=2)

        # Selecionar colunas relevantes
        columns = ['stars', 'review_count', 'latitude', 'longitude', 'categories']
        sampled_df = sampled_df[columns]

        # Exibir os dados carregados
        st.write("### Visualização dos Dados Carregados")
        st.write("Estes são dados de restaurantes do Yelp, com informações como avaliação, localização e categorias.")
        st.dataframe(sampled_df.head()) 

        # Separar variáveis numéricas e categóricas
        numeric_features = ['stars', 'review_count', 'latitude', 'longitude']
        categorical_features = ['categories']

        # Normalizar numéricos e transformar categóricos em One-Hot Encoding
        numeric_transformer = MinMaxScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Aplicar transformação
        processed_data = preprocessor.fit_transform(sampled_df)
        processed_data_dense = processed_data.toarray()

        # Redução de dimensionalidade com PCA (2 componentes)
        pca = PCA(n_components=2, random_state=2)
        processed_data_pca = pca.fit_transform(processed_data_dense)

        # Método do Cotovelo
        st.write("### Método do Cotovelo")
        st.write("O método do cotovelo ajuda a escolher o número ideal de clusters. O ponto de cotovelo é onde a inércia começa a diminuir mais lentamente.")
        inertia = []
        for n in range(1, 11):
            kmeans = KMeans(n_clusters=n, random_state=2)
            kmeans.fit(processed_data)  # Usar processed_data (antes do PCA)
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(8, 4))
        plt.plot(range(1, 11), inertia, marker='o')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Inércia')
        plt.title('Método do Cotovelo')

        # Destacar o ponto de quebra do cotovelo em vermelho
        best_k_elbow = np.argmin(np.diff(inertia, 2)) + 2  
        plt.plot(best_k_elbow, inertia[best_k_elbow-1], 'ro')  

        # Ajustar o tamanho da seta
        plt.annotate('Cotovelo',
                    xy=(best_k_elbow, inertia[best_k_elbow-1]),
                    xytext=(best_k_elbow + 0.5, inertia[best_k_elbow-1] + 0.1 * max(inertia)),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=8))

        st.pyplot(plt)

        # Selecionar número de clusters
        st.write("### Escolha do Número de Clusters")
        n_clusters = st.sidebar.slider("Selecione o número de clusters", 2, 10, 3, 
                            help="Escolha o número de grupos (clusters) para agrupar os restaurantes com base em suas características.")

        # Executar K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=2)
        labels = kmeans.fit_predict(processed_data_pca)

        # Adicionar coluna de clusters ao DataFrame
        sampled_df['cluster'] = labels

        # Método da Silhueta
        with st.expander(f"Ver Método da Silhueta ({n_clusters} clusters)"):
            st.write("### Método da Silhueta")
            st.write("O gráfico da silhueta mostra quão bem cada ponto se encaixa no seu cluster. Valores próximos de 1 indicam uma boa separação.")
            silhouette_avg = silhouette_score(processed_data_pca, labels)
            st.write(f"Média da silhueta para {n_clusters} clusters: **{silhouette_avg:.2f}**")

            # Gráfico da Silhueta
            sample_silhouette_values = silhouette_samples(processed_data_pca, labels)
            fig, ax = plt.subplots(figsize=(8, 6))
            y_lower = 10
            for i in range(n_clusters):
                ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                color = cm.nipy_spectral(float(i) / n_clusters)
                ax.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10
            ax.axvline(x=silhouette_avg, color='red', linestyle='--')
            ax.set_xlabel('Valor da Silhueta')
            ax.set_ylabel('Cluster')
            ax.set_title(f'Silhueta para {n_clusters} clusters')
            st.pyplot(fig)

        # Gráfico de dispersão dos clusters
        with st.expander(f"Ver Gráfico de Dispersão ({n_clusters} clusters)"):
            st.write("### Visualização dos Clusters")
            st.write("Gráfico de dispersão dos clusters após redução de dimensionalidade com PCA.")
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(processed_data_pca[:, 0], processed_data_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title('KMeans Clustering')
            plt.legend(*scatter.legend_elements(), title="Clusters")
            st.pyplot(plt)

        with st.expander("Consultas Por Cluster"):
            # Tabela de médias por cluster
            st.write("### Médias por Cluster")
            st.write("Esta tabela mostra as médias das características numéricas para cada cluster.")
            numeric_columns = sampled_df.select_dtypes(include=[np.number]).columns
            cluster_means = sampled_df.groupby('cluster')[numeric_columns].mean()
            st.write(cluster_means)

            # Mostrar dados de cada cluster
            st.write("### Dados de Cada Cluster")
            selected_cluster = st.selectbox("Selecione um cluster para visualizar:", sampled_df['cluster'].unique())
            st.write(f"#### Dados do Cluster {selected_cluster}:")
            st.dataframe(sampled_df[sampled_df['cluster'] == selected_cluster])

    # CLASSIFICAÇÃO
    elif selecionado == 'Classificação':
        # Título da aplicação
        st.title("Previsão de Faixa de Preço de Restaurantes")

        # Descrição
        st.write("""
        Esta aplicação utiliza modelos de machine learning para prever a faixa de preço de restaurantes com base em suas características.
        Escolha um modelo e insira os valores das features para ver a previsão.
        """)

        # Carregar os modelos salvos e as previsões no conjunto de teste
        modelos = {
            "Regressão Logística (SMOTE)": {
                "modelo": "modelos/modelo_lr_smote.pkl",
                "previsoes": "previsoes/previsoes_lr_smote.pkl",
                "y_test": "test/y_test.pkl"
            },
            "Regressão Logística (SMOTEENN)": {
                "modelo": "modelos/modelo_lr_smoteenn.pkl",
                "previsoes": "previsoes/previsoes_lr_smoteenn.pkl",
                "y_test": "test/y_test.pkl"
            },
            "KNN (SMOTE)": {
                "modelo": "modelos/modelo_knn_smote.pkl",
                "previsoes": "previsoes/previsoes_knn_smote.pkl",
                "y_test": "test/y_test.pkl"
            },
            "KNN (SMOTEENN)": {
                "modelo": "modelos/modelo_knn_smoteenn.pkl",
                "previsoes": "previsoes/previsoes_knn_smoteenn.pkl",
                "y_test": "test/y_test.pkl"
            },
            "XGBoost (SMOTE)": {
                "modelo": "modelos/modelo_xgb_smote.pkl",
                "previsoes": "previsoes/previsoes_xgb_smote.pkl",
                "y_test": "test/y_test.pkl"
            },
            "XGBoost (SMOTEENN)": {
                "modelo": "modelos/modelo_xgb_smoteenn.pkl",
                "previsoes": "previsoes/previsoes_xgb_smoteenn.pkl",
                "y_test": "test/y_test.pkl"
            },
            "Random Forest (SMOTE)": {
                "modelo": "modelos/modelo_rf_smote.pkl",
                "previsoes": "previsoes/previsoes_rf_smote.pkl",
                "y_test": "test/y_test.pkl"
            },
            "Random Forest (SMOTEENN)": {
                "modelo": "modelos/modelo_rf_smoteenn.pkl",
                "previsoes": "previsoes/previsoes_rf_smoteenn.pkl",
                "y_test": "test/y_test.pkl"
            }
        }

        # Selecionar o modelo
        modelo_selecionado = st.selectbox("Escolha um modelo:", list(modelos.keys()))

        # Carregar o modelo, previsões e y_test
        modelo = joblib.load(modelos[modelo_selecionado]["modelo"])
        previsoes = joblib.load(modelos[modelo_selecionado]["previsoes"])
        y_test = joblib.load(modelos[modelo_selecionado]["y_test"])

        # Inputs para as features
        st.sidebar.header("Insira as características do restaurante")

        def user_input_features():
            stars = st.sidebar.slider('Avaliação (stars)', 1.0, 5.0, 3.5, step=0.5)
            review_count = st.sidebar.slider('Número de avaliações (review_count)', 0, 1000, 100)
            latitude = st.sidebar.number_input('Latitude', value=37.7749)
            longitude = st.sidebar.number_input('Longitude', value=-122.4194)
            alcohol = st.sidebar.selectbox('Serviço de álcool (Alcohol)', ['Nenhum', 'Beer&Wine', 'Full_Bar'])
            bike_parking = st.sidebar.selectbox('Estacionamento para bicicletas (BikeParking)', ['Sim', 'Não'])
            delivery = st.sidebar.selectbox('Entrega (RestaurantsDelivery)', ['Sim', 'Não'])
            takeout = st.sidebar.selectbox('Takeout (RestaurantsTakeOut)', ['Sim', 'Não'])

            # Mapear inputs para valores numéricos
            alcohol_map = {'Nenhum': 0, 'Beer&Wine': 1, 'Full_Bar': 2}
            bike_parking_map = {'Sim': 1, 'Não': 0}
            delivery_map = {'Sim': 1, 'Não': 0}
            takeout_map = {'Sim': 1, 'Não': 0}

            data = {
                'stars': stars,
                'review_count': review_count,
                'latitude': latitude,
                'longitude': longitude,
                'Alcohol': alcohol_map[alcohol],
                'BikeParking': bike_parking_map[bike_parking],
                'RestaurantsDelivery': delivery_map[delivery],
                'RestaurantsTakeOut': takeout_map[takeout]
            }

            features = pd.DataFrame(data, index=[0])
            return features
        
        with st.expander(f"Relatório e Matriz {modelo_selecionado}"):
            # Relatório de Classificação
            st.subheader(f"Relatório de Classificação do {modelo_selecionado}")
            report = classification_report(y_test, previsoes)
            st.markdown(f"```\n{report}\n```")

            # Exibir a matriz de confusão
            st.subheader('Matriz de Confusão:')
            cm = confusion_matrix(y_test, previsoes)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Baixo', 'Médio', 'Alto'], yticklabels=['Baixo', 'Médio', 'Alto'])
            plt.xlabel('Previsão')
            plt.ylabel('Real')
            plt.title(f'Matriz de Confusão - {modelo_selecionado}')
            st.pyplot(plt)

        # Coletar inputs do usuário
        input_df = user_input_features()

        # Exibir as features inseridas
        st.subheader('Características inseridas:')
        st.write(input_df)

        # Fazer a previsão
        if st.button("Prever"):
            prediction = modelo.predict(input_df)
            prediction_proba = modelo.predict_proba(input_df)

            # Mapear a previsão para o nome da classe
            classes = ['Baixo', 'Médio', 'Alto']
            predicted_class = classes[prediction[0]]

            # Exibir a previsão
            st.subheader('Previsão:')
            st.write(f"A faixa de preço prevista é: **{predicted_class}**")

            # Exibir probabilidades
            st.subheader('Probabilidades:')
            for i, prob in enumerate(prediction_proba[0]):
                st.write(f"Probabilidade de {classes[i]}: {prob:.2f}")



if __name__ == '__main__':
    main()