"""
# Tech Challenge Fase 4
# Data Analytics FIAP - 02/2025
# Alison Sérgio de Amarins Germano - RM 357521

*Ambiente de desenvolvimento utilizado foi o VS CODE

**1. Desenvolver um modelo preditivo com dados da Cotação do preço de barril de petróleo para criar uma série temporal e prever o fechamento.**

**1. Instalando Bibliotecas**
"""
# Importando e Aplicando o álias a cada biblioteca
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, SeasonalWindowAverage, AutoARIMA
from statsforecast.utils import ConformalIntervals
from prophet import Prophet
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from statsmodels.tsa.statespace.sarimax import SARIMAX
from IPython.display import Image
import altair as alt
import seaborn as sns
# Ignorar avisos específicos emitidos pelo Pandas tipo SettingWithCopyWarning
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore")

st.write("# Tech Challenge Fase 4")
st.write('<b><span style="font-size: 20px;">Alison Sérgio de Amarins Germano - RM 357521</b>', unsafe_allow_html=True) 
st.write("# Objetivos : ")
'''

Dashboard com um storytelling que traga insights relevantes sobre a variação do preço do petróleo.

Modelo de Machine Learning que faça a previsão do preço do petróleo.

Plano para fazer o deploy em produção do modelo, com as ferramentas que são necessárias.

Um vídeo de até 5 (cinco) minutos explicando todo o desenvolvimento do seu projeto.

'''
st.write('<b><span style="font-size: 20px;"></b>', unsafe_allow_html=True) 
#"""**1**.Acessando a base de dados"""
# Títulos
ticker = "BZ=F"
data_principal = yf.download(ticker, start="2014-09-19", end="2024-09-20")
# Resetar o índice para mover o DatetimeIndex para uma coluna
data_principal2 = data_principal.copy()
data_principal2.reset_index(inplace=True)
# Selecionar e renomear as colunas para a nova estrutura
df_novo = pd.DataFrame({
    ('Price', 'ds'): data_principal2['Date'],  # A coluna de datas
    ('Price', 'y'): data_principal2['Close'],  # A coluna de fechamento
    ('Price', 'unique_id'): ticker  # Adicionar o ticker como valor fixo
})
df_novo2 = df_novo.copy()
# Segmentando o DataFrame
dados = data_principal[["Open", "Close"]]
dados.dropna(inplace=True)
dados.reset_index(inplace=True)
# Título da aplicação no navegador
st.title("Visualização dos Dados Históricos")
st.write("Gráfico de evolução do fechamento dos dados históricos.")
# Criando a visualização com matplotlib
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(dados["Date"], dados["Close"], label="Evolução Fechamento")
ax.set_title("Dados Históricos")
ax.set_xlabel("Período")
ax.set_ylabel("Quantidade de Pontos")
ax.legend()
# Renderizar o gráfico no navegador usando Streamlit
st.pyplot(fig)
st.write("Exibindo os últimos valores de abertura e fechamento")
st.write(dados.head())  # Mostra as primeiras linhas do DataFrame
"""**Analisando Outliers**"""
dados2 = dados.copy()
dados2.columns = ['Date', 'Open', 'Close']
# Criando o boxplot
st.title("Visualização de Boxplot")
st.write("Este gráfico mostra a distribuição dos valores de fechamento (Close).")
# Criando o boxplot com Seaborn
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(x="Close", data=dados2, ax=ax)
ax.set_title("Boxplot")
st.pyplot(fig)
"""**Mesclar o Gráfico Violin com Box Plot**"""
st.title("Box Plot e Violin Plot")
st.write("Este gráfico mescla um **Violin Plot** e um **Box Plot** para identificar valores discrepantes.")
# Criando o gráfico
fig, ax = plt.subplots(figsize=(12, 8))
# Violin Plot (em cinza claro)
sns.violinplot(x="Close", data=dados2, ax=ax, color="lightgray")
# Box Plot (em azul escuro)
sns.boxplot(x="Close", data=dados2, ax=ax, whis=1.5, color="darkblue")
ax.set_title("Visualização Box Plot e Violin Plot")
st.pyplot(fig)
"""Observa-se que não existem dados discrepantes.

**Histograma dos Pontos de Fechamento**
"""
st.title("Histograma dos Pontos de Fechamento")
st.write("Este gráfico mostra o histograma dos pontos de fechamento ('Close').")
# Criando o histograma
fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(dados["Close"], bins=20, edgecolor="black")
# Adicionando título e rótulos aos eixos
ax.set_title("Histograma dos Pontos de Fechamento")
ax.set_xlabel("Pontos de Fechamento")
ax.set_ylabel("Frequência")
st.pyplot(fig)
"""***Aplicando a Amplitude  nos Valores***"""
dadosRaiz = dados.copy()
dadosRaiz["Close_Raiz"] = np.sqrt(dadosRaiz["Close"])
st.title("Histograma dos Pontos de Fechamento Ampliado")
st.write("Este gráfico mostra o histograma dos pontos de fechamento ampliado ('Close_Raiz').")
# Criando o histograma
fig, ax = plt.subplots(figsize=(12, 8))
# Plotando o histograma
ax.hist(dadosRaiz["Close_Raiz"], bins=20, edgecolor="black")
# Adicionando título e rótulos aos eixos
ax.set_title("Histograma dos Pontos de Fechamento Ampliado")
ax.set_xlabel("Pontos de Fechamento Ampliado")
ax.set_ylabel("Frequência")
# Exibindo o gráfico no Streamlit
st.pyplot(fig)
"""**Gráfico de Pontos de Abertura e Fechamento**"""
import matplotlib.pyplot as plt
# Plotar a variação de Open e Close ao longo do tempo
data_principal.reset_index(inplace=True)
st.title("Gráfico Open vs Close")
st.write("Este gráfico mostra a comparação entre os pontos de abertura ('Open') e fechamento ('Close').")
# Criando o gráfico
fig, ax = plt.subplots(figsize=(12, 8))
# Plotando os dados de "Open"
ax.plot(data_principal["Date"], data_principal["Open"], label="Open", color='red', linewidth=0.3)
# Plotando os dados de "Close"
ax.plot(data_principal["Date"], data_principal["Close"], label="Close", color='blue', linewidth=0.2)
ax.set_title("Open vs Close")
ax.set_xlabel("Data")
ax.set_ylabel("Pontos")
ax.legend()
st.pyplot(fig)
"""**Identificando o Padrão da Série Temporal**"""
result = seasonal_decompose(dados["Close"], model="multiplicative", period=252)
result.seasonal.iloc[:252].plot(figsize=(12, 8))
"""**Realizando a decomposição de série temporal para separar os componentes mais simples e interpretáveis, com o objetivo de analisar o comportamento e melhorar a qualidade dos modelo preditivo.**"""
# Decomposição da série temporal
# Tendência - Direção
# Sazonalidade - recorrência das oscilações
# Resíduo - o q sobra do sinal
df = data_principal[["Date", "Close"]]
df.set_index("Date",inplace=True)
df.index = pd.to_datetime(df.index)
df = df.asfreq('D')
df['Close'] = df['Close'].ffill()
resultados = seasonal_decompose(df,  period=252)
st.title("Decomposição de Séries Temporais")

# Descrição
st.write("Este gráfico mostra a decomposição da série temporal em componentes observados, tendência, sazonalidade e resíduos.")
# Criando os subgráficos
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
# Plotando o componente 'Observed'
resultados.observed.plot(ax=ax1, label='Observed')
ax1.legend()
# Plotando o componente 'Trend'
resultados.trend.plot(ax=ax2, label='Trend')
ax2.legend()
# Plotando o componente 'Seasonal'
resultados.seasonal.plot(ax=ax3, label='Seasonal')
ax3.legend()
# Plotando o componente 'Resid'
resultados.resid.plot(ax=ax4, label='Resid')
ax4.legend()
# Ajustando o layout
plt.tight_layout()
# Exibindo o gráfico no Streamlit
st.pyplot(fig)
"""**Distribuição dos Resíduos com Histograma e Teste de Normalidade.**"""
# Criando novos gráficos para os resíduos: histograma e teste de normalidade
st.title("Análise de Resíduos")
st.write("Este gráfico mostra o histograma dos resíduos e o teste de normalidade com o QQ plot.")
# Criando os subgráficos
fig, (ax5, ax6) = plt.subplots(2, 1, figsize=(12, 8))
# Plotando o histograma dos resíduos
sns.histplot(resultados.resid.dropna(), kde=True, ax=ax5)
ax5.set_title('Histograma dos Resíduos')
# Teste de normalidade: QQ plot
qqplot(resultados.resid.dropna(), line='s', ax=ax6)
ax6.set_title('Normalidade')
# Ajustando o layout
plt.tight_layout()
# Exibindo o gráfico no Streamlit
st.pyplot(fig)
# Teste Shapiro-Wilk para normalidade
stat, p_value = shapiro(resultados.resid.dropna())
st.write(f'Shapiro-Wilk Teste de Normalidade: p-value = {p_value:.4f}')
"""**O p-value = 0.0000 logo pode-se concluir que os resíduos do modelo não seguem uma distribuição normal.**

**Identificar se é uma Série Estacionaria**
"""
# Estacionária ou não estacionária
# ADF - Augmented Dickey-Fuller
# H0 - Hipótese Nula (não é estacionária)
# H1 ou HA (Depende da abordagem) - Hipótese Alternativa (rejeição da hipótese nula)
# p -value = 0.05 (5%), então rejeitamos H0 com um nível de confiança de 95%
sns.set_style('darkgrid')
x = df.Close.values
result = adfuller(x)
st.write("Teste ADF")
st.write(f"Teste Estatístico: {result[0]}")
st.write(f"P-Value: {result[1]}")
st.write("Valores críticos:")
for key, value in result[4].items():
  st.write(f"\t{key}: {value}")
st.write(f"**P-Value:** {p_value:.4f}, Teste Estatístico:  {result[0]}  maior que os valores críticos, indicando que não é uma série estacionária, então fazer a transformação para estacionária, mas antes um pouco mais de análise gráfica.")
# Plotando a linha da média móvel em cima da série
# 252 dias úteis
ma = df.rolling(252).mean()
st.title("Visualização de Séries Temporais")
# Criando a figura e o eixo
fig, ax = plt.subplots(figsize=(12, 8))
df.plot(ax=ax, legend=False)
ma.plot(ax=ax, legend=False, color='r')
# Ajustando o layout
plt.tight_layout()
# Exibindo o gráfico no Streamlit
st.pyplot(fig)
"""Aplicando a função logarítimica Numpy para suavizar a série realizando uma transformação da escala tentanto ver se fica mais estacionária."""
df_log = np.log(df)
ma_log = df_log.rolling(252).mean()
# Criando a figura e o eixo
fig, ax = plt.subplots(figsize=(12, 8))
# Plotando os dados
df_log.plot(ax=ax, legend=False)
ma_log.plot(ax=ax, legend=False, color='r')
# Ajustando o layout
plt.tight_layout()
# Exibindo o gráfico no Streamlit
st.pyplot(fig)
"""Subtraindo a média móvel e utilizar 252 dias."""
df_s = (df_log - ma_log).dropna()
ma_s = df_s.rolling(252).mean()
std = df_s.rolling(252).std()
# Criando a figura e o eixo
fig, ax = plt.subplots(figsize=(12, 8))
# Plotando os dados
df_s.plot(ax=ax, legend=False, label="Série Original")
ma_s.plot(ax=ax, legend=False, color='r', label="Média Móvel")
std.plot(ax=ax, legend=False, color='g', label="Desvio Padrão")
# Ajustando o layout
plt.tight_layout()
# Exibindo o gráfico no Streamlit
st.pyplot(fig)
"""**Realizar o Teste ADF Novamente.**"""
x_s = df_s.Close.values
result_s = adfuller(x_s)
st.write("Teste ADF")
st.write(f"Teste Estatístico: {result_s[0]}")
st.write(f"P-Value: {result_s[1]}")
st.write("Valores críticos:")

for key, value in result_s[4].items():
  st.write(f"\t{key}: {value}")
st.write(f"**P-Value:** {result_s[1]}, Teste Estatístico: {result_s[0]}  melhorou sendo menor em 5% e 10% dos valores críticos, indicando que a série se aproximou muito de uma estacionária.")
"""
**Aplicando a primeira diferenciação e o teste ADF**
"""
df_diff = df_s.diff(1)
ma_diff = df_diff.rolling(252).mean()
std_diff = df_diff.rolling(252).std()
fig, ax = plt.subplots(figsize=(12, 8))
# Plotando os dados
df_diff.plot(ax=ax, legend=False, label="Série Diferenciada")
ma_diff.plot(ax=ax, legend=False, color='r', label="Média Móvel")
std_diff.plot(ax=ax, legend=False, color='g', label="Desvio Padrão")
# Ajustando o layout
plt.tight_layout()
# Exibindo o gráfico no Streamlit
st.pyplot(fig)

x_diff = df_diff.Close.dropna().values
result_diff = adfuller(x_diff)
st.write("Teste ADF")
st.write(f"Teste Estatístico: {result_diff[0]}")
st.write(f"P-Value: {result_diff[1]}")
st.write("Valores críticos:")

for key, value in result_diff[4].items():
  st.write(f"\t{key}: {value}")

"""**P-Value exponencial muito próximo a zero e o Teste Estatístico menor que os Valores Críticos, então se transformou em estacionária.**

**Aplicando a Função de Autocorrelação ACF e Autocorrelação Parcial PACF e Analisando**
"""
# Utilizando o DF suavizado
df_diff.info()
"""**Realizando a análise de Auto Correlação para 21 dias.**"""
# nlags esta em dias podendo ser ajustado
lag_acf = acf(df_diff.Close.dropna(), nlags=21)
lag_pacf = pacf(df_diff.Close.dropna(), nlags=21)

# Observando as diferenças ACF e PACF
# Valor -1 porquê a série foi diferenciada uma vez.
# ACF
# Título da aplicação
st.title("Análise de ACF e PACF")
# Criando a figura para ACF
fig_acf, ax_acf = plt.subplots(figsize=(12, 6))
ax_acf.plot(lag_acf, label="ACF")
ax_acf.axhline(y=0, linestyle='--', color='gray', linewidth=0.5)
ax_acf.axhline(y=-1.96 / np.sqrt(len(df_diff.Close) - 1), linestyle='--', color='gray', linewidth=0.5)
ax_acf.axhline(y=1.96 / np.sqrt(len(df_diff.Close) - 1), linestyle='--', color='gray', linewidth=0.5)
ax_acf.set_title("Autocorrelation Function (ACF)")
st.pyplot(fig_acf)
# Criando a figura para PACF
fig_pacf, ax_pacf = plt.subplots(figsize=(12, 6))
ax_pacf.plot(lag_pacf, label="PACF")
ax_pacf.axhline(y=0, linestyle='--', color='gray', linewidth=0.5)
ax_pacf.axhline(y=-1.96 / np.sqrt(len(df_diff.Close.dropna()) - 1), linestyle='--', color='gray', linewidth=0.5)
ax_pacf.axhline(y=1.96 / np.sqrt(len(df_diff.Close.dropna()) - 1), linestyle='--', color='gray', linewidth=0.5)
ax_pacf.set_title("Partial Autocorrelation Function (PACF)")
st.pyplot(fig_pacf)
"""Ilustrando melhor a identificação dos valores x e y"""
# Valor de x
#Image('/content/acf_x.jpg')
st.image("acf_x.jpg", caption='ACF')
#Image('/content/pacf_y.jpg')
st.image("pacf_y.jpg", caption='PACF')

#Analisando o quanto o período de 21 dias está relacionado com outro
# ACF Relação Direta e Indireta
# PACF Relação apenas direta
# Filtrar os últimos 21 dias
df_filtered = df.loc[df.index >= (df.index.max() - pd.Timedelta(days=21)), 'Close']
# Criando a figura para ACF
fig_acf, ax_acf = plt.subplots(figsize=(12, 6))
plot_acf(df_filtered, ax=ax_acf)
ax_acf.set_title("Autocorrelation Function (ACF) - Últimos 21 Dias")
st.pyplot(fig_acf)
# Criando a figura para PACF
fig_pacf, ax_pacf = plt.subplots(figsize=(12, 6))
plot_pacf(df_filtered, ax=ax_pacf)
ax_pacf.set_title("Partial Autocorrelation Function (PACF) - Últimos 21 Dias")
st.pyplot(fig_pacf)
st.write('---------------------------------------------------------------------------------------------------')
# Remover MultiIndex para acessar colunas diretamente
df_novo2.columns = [col[1] if col[1] else col[0] for col in df_novo2.columns]
# Verificar as colunas
# Garantir que as colunas necessárias estejam presentes
colunas_necessarias = ['ds', 'y', 'unique_id']
# Filtrar datas de 6 meses
dt_inicial = '2024-03-19'
dt_final = '2024-09-19'
df_filtrado = df_novo2[(df_novo2['ds'] >= dt_inicial) & (df_novo2['ds'] <= dt_final)]
# Separar treino e validação
tamanho = int(len(df_filtrado) * 0.8)
treino = df_filtrado.iloc[:tamanho]
valid = df_filtrado.iloc[tamanho:]
h = valid['ds'].nunique()
# Treinando o modelo
model = StatsForecast(models=[Naive()], freq='D', n_jobs=-1)
model.fit(treino[['ds', 'unique_id', 'y']])
# Previsão
forecast_df = model.predict(h=h, level=[90])
forecast_df = forecast_df.reset_index().merge(valid[['ds', 'unique_id', 'y']], on=['ds', 'unique_id'], how='left').dropna()
# Métricas
y_true = forecast_df['y'].values
y_pred = forecast_df['Naive'].values
#-----------------------------------------------------------------------------------
#Altair NAIVE
st.title("Previsão com Modelo Naive")
# Criar DataFrame consolidado para Altair
df_treino = pd.DataFrame({'ds': treino['ds'], 'y': treino['y'], 'tipo': 'Treino'})
df_validacao = pd.DataFrame({'ds': forecast_df['ds'], 'y': forecast_df['y'], 'tipo': 'Validação (Real)'})
df_previsao = pd.DataFrame({'ds': forecast_df['ds'], 'y': forecast_df['Naive'], 'tipo': 'Previsão (Naive)'})
# Concatenar os DataFrames
df_total = pd.concat([df_treino, df_validacao, df_previsao])
# Criar gráfico com Altair
chart = alt.Chart(df_total).mark_line().encode(
    x='ds:T',
    y='y:Q',
    color='tipo:N',
    strokeDash=alt.condition(
        alt.datum.tipo == 'Previsão (Naive)',
        alt.value([5, 5]),  # Linhas tracejadas para previsão
        alt.value([0])       # Linhas normais para treino e validação
    )
).properties(
    width=800,
    height=400,
    title="Naive"
)
# Exibir no Streamlit
st.altair_chart(chart, use_container_width=True)
#-----------------------------------------------------------------------------------
def metricas(y_true, y_pred):
    """
    Calcula várias métricas de avaliação de séries temporais:
    WMAPE, MAE, MSE, RMSE, MAPE, SMAPE, MASE.

    Parâmetros:
    y_true: Valores reais (array ou lista).
    y_pred: Valores previstos (array ou lista).

    Retorno: Dicionário contendo as métricas calculadas.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # WMAPE - Erro Percentual Absoluto Ponderado
    wmape_value = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

    # MAE - Erro Absoluto Médio
    mae = np.mean(np.abs(y_true - y_pred))

    # MSE - Erro Quadrático Médio
    mse = np.mean((y_true - y_pred) ** 2)

    # RMSE - Raiz do Erro Quadrático Médio
    rmse = np.sqrt(mse)

    # MAPE - Erro Percentual Absoluto Médio
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # SMAPE - Erro Percentual Absoluto Médio Simétrico
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

    # MASE - Erro Médio Absoluto Escalado
    naive_forecast = np.roll(y_true, shift=1)  # Previsão naive (y_t-1)
    naive_forecast[0] = np.nan  # O primeiro valor fica inválido
    mase = np.nanmean(np.abs(y_true[1:] - y_pred[1:]) / np.abs(y_true[1:] - naive_forecast[1:]))

    return {
        'WMAPE': wmape_value,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape,
        'MASE': mase
    }
resultado_metricas = metricas(y_true, y_pred)
st.write("### Métricas de Avaliação")
for metric, value in resultado_metricas.items():
    st.write(f"{metric}: {value:.4f}")
# Criando DataFrame para armazenar os resultados
resultados_modelos = pd.DataFrame(columns=['Modelo', 'WMAPE', 'MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'MASE'])
#-----------------------------------------------------------------------------------
st.title("Previsão com Modelo SeasonalNaive")
# n_jobs -1 parâmetro padrão, para utilizar as CPUS
# season_length=5 Dias úteis na semana
model_s = StatsForecast(models=[SeasonalNaive(season_length=21)], freq='D', n_jobs=-1)
model_s.fit(treino)
# Convertendo para datetime o índice
valid['ds'] = pd.to_datetime(valid['ds'])
forecast_dfs = model_s.predict(h=h, level=[90])
forecast_dfs = forecast_dfs.reset_index().merge(valid, on=['ds', 'unique_id'], how='left').dropna()
# Remover a coluna 'index' do DataFrame
forecast_dfs = forecast_dfs.drop(columns=['index'])
df_treino = pd.DataFrame({'ds': treino['ds'], 'y': treino['y'], 'tipo': 'Treino'})
df_validacao = pd.DataFrame({'ds': forecast_dfs['ds'], 'y': forecast_dfs['y'], 'tipo': 'Validação (Real)'})
df_previsao = pd.DataFrame({'ds': forecast_dfs['ds'], 'y': forecast_dfs['SeasonalNaive'], 'tipo': 'Previsão (SeasonalNaive)'})
# Concatenar os DataFrames
df_total = pd.concat([df_treino, df_validacao, df_previsao])
# Criar gráfico com Altair
chart = alt.Chart(df_total).mark_line().encode(
    x='ds:T',
    y='y:Q',
    color='tipo:N',
    strokeDash=alt.condition(
        alt.datum.tipo == 'Previsão (SeasonalNaive)',
        alt.value([5, 5]),  # Linhas tracejadas para previsão
        alt.value([0])       # Linhas normais para treino e validação
    )
).properties(
    width=800,
    height=400,
    title="SeasonalNaive"
)
# Exibir no Streamlit
st.altair_chart(chart, use_container_width=True)
#
resultado_metricas = metricas(forecast_dfs['y'].values,  forecast_dfs['SeasonalNaive'].values )
for metric, value in resultado_metricas.items():
    st.write(f"{metric}: {value:.4f}")
modelo = 'SeasonalNaive'
novo_resultado = pd.DataFrame([{'Modelo': modelo, **resultado_metricas}])
resultados_modelos = pd.concat([resultados_modelos, novo_resultado], ignore_index=True)
#-----------------------------------------------------------
st.title("Previsão com SeasonalWindowAverage")
"""**Utilizando o modelo com Média Móvel com Sazonalidade o SeasonalWindowAverage.**"""
# n_jobs -1 parâmetro padrão, para utilizar as CPUS
# Adicionado o parâmetro window_size que são as janelas, posso usar 2 para pegar a média das 2 últimas semanas
# season_length=5 , utilizado 5 dias úteis na semana
model_sm = StatsForecast(models=[SeasonalWindowAverage(season_length=5, window_size=2, prediction_intervals=ConformalIntervals(h=h, n_windows=2))], freq='D', n_jobs=-1)
model_sm.fit(treino)
#  Convertendo para datetime o índice
valid['ds'] = pd.to_datetime(valid['ds'])
forecast_dfsm = model_sm.predict(h=h, level=[90])
forecast_dfsm = forecast_dfsm.reset_index().merge(valid, on=['ds', 'unique_id'], how='left').dropna()
# Remover a coluna 'index' do DataFrame
forecast_dfsm = forecast_dfsm.drop(columns=['index'])
# Criar DataFrame consolidado para Altair
df_treino = pd.DataFrame({'ds': treino['ds'], 'y': treino['y'], 'tipo': 'Treino'})
df_validacao = pd.DataFrame({'ds': forecast_dfsm['ds'], 'y': forecast_dfsm['y'], 'tipo': 'Validação (Real)'})
df_previsao = pd.DataFrame({'ds': forecast_dfsm['ds'], 'y': forecast_dfsm['SeasWA'], 'tipo': 'Previsão (SeasWA)'})
# Concatenar os DataFrames
df_total = pd.concat([df_treino, df_validacao, df_previsao])
# Criar gráfico com Altair
chart = alt.Chart(df_total).mark_line().encode(
    x='ds:T',
    y='y:Q',
    color='tipo:N',
    strokeDash=alt.condition(
        alt.datum.tipo == 'Previsão (SeasWA)',
        alt.value([5, 5]),  # Linhas tracejadas para previsão
        alt.value([0])       # Linhas normais para treino e validação
    )
).properties(
    width=800,
    height=400,
    title="SeasWA"
)
# Exibir no Streamlit
st.altair_chart(chart, use_container_width=True)
resultado_metricas = metricas(forecast_dfsm['y'].values,  forecast_dfsm['SeasWA'].values )
for metric, value in resultado_metricas.items():
    st.write(f"{metric}: {value:.4f}")
modelo = 'SeasWA'
novo_resultado = pd.DataFrame([{'Modelo': modelo, **resultado_metricas}])
resultados_modelos = pd.concat([resultados_modelos, novo_resultado], ignore_index=True)
#------------------------------------------------------------------------------------------------------
st.title("Previsão com AUTOARIMA")
"""**Utilizando o modelo AUTOARIMA, olha o fechamento no passado e encontra uma correlação futura, semelhante ao modelo Base Line Naive.**"""

#ARIMA
#(AR) : Autoregressivo, I: Integrado, MA: Média Móvel

#A(x,y,z) => ACF, PACF
# Determinando o valor de x

# Considera 3 pontos:
# Modelo Autoregressivo
# Modelo Integrado
# Média Móvel
# I - quantidade que a série foi diferenciada
model_a = StatsForecast(models=[AutoARIMA(season_length=5)], freq='D', n_jobs=-1)
model_a.fit(treino)
valid['ds'] = pd.to_datetime(valid['ds'])
forecast_dfa = model_a.predict(h=h, level=[90])
forecast_dfa = forecast_dfa.reset_index().merge(valid, on=['ds', 'unique_id'], how='left').dropna()
# Remover a coluna 'index' do DataFrame
forecast_dfa = forecast_dfa.drop(columns=['index'])
# Criar DataFrame consolidado para Altair
df_treino = pd.DataFrame({'ds': treino['ds'], 'y': treino['y'], 'tipo': 'Treino'})
df_validacao = pd.DataFrame({'ds': forecast_dfa['ds'], 'y': forecast_dfa['y'], 'tipo': 'Validação (Real)'})
df_previsao = pd.DataFrame({'ds': forecast_dfa['ds'], 'y': forecast_dfa['AutoARIMA'], 'tipo': 'Previsão (AutoARIMA)'})
# Concatenar os DataFrames
df_total = pd.concat([df_treino, df_validacao, df_previsao])
# Criar gráfico com Altair
chart = alt.Chart(df_total).mark_line().encode(
    x='ds:T',
    y='y:Q',
    color='tipo:N',
    strokeDash=alt.condition(
        alt.datum.tipo == 'Previsão (AutoARIMA)',
        alt.value([5, 5]),  # Linhas tracejadas para previsão
        alt.value([0])       # Linhas normais para treino e validação
    )
).properties(
    width=800,
    height=400,
    title="AutoARIMA"
)
st.altair_chart(chart, use_container_width=True)
resultado_metricas = metricas(forecast_dfa['y'].values,  forecast_dfa['AutoARIMA'].values )
for metric, value in resultado_metricas.items():
    st.write(f"{metric}: {value:.4f}")

modelo = 'AutoARIMA'
novo_resultado = pd.DataFrame([{'Modelo': modelo, **resultado_metricas}])
resultados_modelos = pd.concat([resultados_modelos, novo_resultado], ignore_index=True)
#---------------------------------------------------------------------------------
"""**Utilizando o Modelo PROPHET**"""
st.title("Prophet")
df_prophet = df_filtrado
# Ajustar / Treinar o modelo
model_prophet = Prophet(daily_seasonality=True)
model_prophet.fit(df_prophet)
# freq='B' somente dias úteis
future = model_prophet.make_future_dataframe(periods=5, freq='B')
forecast_prophet = model_prophet.predict(future)
# Merge para combinar valores reais e predições
df_merged = pd.merge(df_prophet, forecast_prophet[['ds', 'yhat']], on='ds', how='inner')
# Criar DataFrame consolidado para Altair
df_real = pd.DataFrame({'ds': df_prophet['ds'], 'y': df_prophet['y'], 'tipo': 'Valores Reais'})
df_previsao = pd.DataFrame({'ds': forecast_prophet['ds'], 'y': forecast_prophet['yhat'], 'tipo': 'Previsão Prophet'})
df_incerteza = pd.DataFrame({
    'ds': forecast_prophet['ds'],
    'yhat_lower': forecast_prophet['yhat_lower'],
    'yhat_upper': forecast_prophet['yhat_upper']
})
# Concatenar os DataFrames para um gráfico único
df_total = pd.concat([df_real, df_previsao])
# Criar gráfico de linha para os valores reais e previsão
chart = alt.Chart(df_total).mark_line().encode(
    x='ds:T',
    y='y:Q',
    color='tipo:N'
).properties(
    width=800,
    height=400,
    title="Previsão usando Prophet"
)
# Criar a faixa de incerteza (área sombreada)
band = alt.Chart(df_incerteza).mark_area(opacity=0.3, color="green").encode(
    x='ds:T',
    y='yhat_lower:Q',
    y2='yhat_upper:Q'
)
st.altair_chart(band + chart, use_container_width=True)
resultado_metricas = metricas(df_merged['y'].values,  df_merged['yhat'].values )
for metric, value in resultado_metricas.items():
    st.write(f"{metric}: {value:.4f}")
modelo = 'Prophet'
novo_resultado = pd.DataFrame([{'Modelo': modelo, **resultado_metricas}])
resultados_modelos = pd.concat([resultados_modelos, novo_resultado], ignore_index=True)
"""**Utilizando o Modelo XGBOOST**"""
st.title("Previsões do XGBoost vs. Valores Reais")
# Criando cópia do DataFrame
df_xgb = df_filtrado.copy()
df_xgb['ds'] = pd.to_datetime(df_xgb['ds'])
# Extraindo características temporais
df_xgb['ano'] = df_xgb['ds'].dt.year
df_xgb['mes'] = df_xgb['ds'].dt.month
df_xgb['dia'] = df_xgb['ds'].dt.day
df_xgb['diadasemana'] = df_xgb['ds'].dt.dayofweek
# Definindo as features X e o alvo y
X = df_xgb[['ano', 'mes', 'dia']]
y = df_xgb['y']
# Proporção de treino (80%)
split_index = int(len(X) * 0.8)
# Dividindo os dados manualmente
X_train, X_valid = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_valid = y.iloc[:split_index], y.iloc[split_index:]
# Convertendo para o formato DMatrix do XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid)
# Definição dos hiperparâmetros
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 1000
}
# Treinando o modelo diretamente sem usar sklearn
xgb_model = xgb.train(params, dtrain, num_boost_round=100)
# Fazendo previsões
y_pred = xgb_model.predict(dvalid)
# Criar DataFrame consolidado para Altair
df_real = pd.DataFrame({'ds': df_xgb['ds'].iloc[split_index:], 'y': y_valid, 'tipo': 'Valores Reais'})
df_previsao = pd.DataFrame({'ds': df_xgb['ds'].iloc[split_index:], 'y': y_pred, 'tipo': 'Previsões XGBoost'})
# Concatenar os DataFrames
df_total = pd.concat([df_real, df_previsao])
# Criar gráfico de linha para os valores reais e previsão
chart = alt.Chart(df_total).mark_line().encode(
    x=alt.X('ds:T', axis=alt.Axis(title='Data', format='%Y-%m-%d')),
    y=alt.Y('y:Q', axis=alt.Axis(title='Valores')),
    color='tipo:N'
).properties(
    width=800,
    height=400,
    title="Previsões do XGBoost vs. Valores Reais"
)
# Exibir no Streamlit
st.altair_chart(chart, use_container_width=True)
resultado_metricas = metricas(y_valid, y_pred)
for metric, value in resultado_metricas.items():
    st.write(f"{metric}: {value:.4f}")
modelo = 'XGBoost'
novo_resultado = pd.DataFrame([{'Modelo': modelo, **resultado_metricas}])
resultados_modelos = pd.concat([resultados_modelos, novo_resultado], ignore_index=True)
st.title("Resultado dos Modelos")
st.write(resultados_modelos)
st.title("Conclusão")
'''

Se considerarmos todas as métricas, o modelo "SeasWA" parece ter um desempenho geral melhor, pois apresenta os menores valores de MAE, RMSE e MASE, que são métricas críticas para avaliação da previsão.

Se focarmos apenas em WMAPE, o Prophet seria o melhor.

Se analisarmos MAPE e SMAPE, o XGBoost se destaca.

Porém, como o SeasWA tem os melhores resultados em várias métricas importantes, ele parece ser a melhor escolha globalmente
'''
