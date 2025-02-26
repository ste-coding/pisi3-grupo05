# PISI3 Grupo 05

## Descrição
Este projeto é uma aplicação de análise de dados e aprendizado de máquina que utiliza dados do Yelp para realizar clustering, classificação e análise exploratória de dados (EDA).

## Estrutura do Projeto
- `streamlit/`: Contém os arquivos para a interface do usuário utilizando Streamlit.
  - `st.py`: Script principal do Streamlit.
  - `app.py`: Script adicional do Streamlit.
- `machine-learning/`: Contém notebooks de aprendizado de máquina.
  - `clustering.ipynb`: Notebook para clustering.
  - `classification_2.ipynb`: Notebook para classificação.
- `dataset/`: Contém o dataset utilizado no projeto.
  - `yelp_academic_dataset_business_cleaned.csv`: Dataset do Yelp.

## Instalação
1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/pisi3-grupo05.git
   cd pisi3-grupo05
   ```

2. Crie um ambiente virtual e ative-o:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows use `venv\Scripts\activate`
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Uso
1. Execute a aplicação Streamlit:
   ```bash
   streamlit run streamlit/st.py
   ```

2. Abra os notebooks de aprendizado de máquina para explorar as análises:
   - `machine-learning/clustering.ipynb`
   - `machine-learning/classification_2.ipynb`

## Contribuição
Sinta-se à vontade para abrir issues e pull requests. Toda contribuição é bem-vinda!

## Licença
Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
