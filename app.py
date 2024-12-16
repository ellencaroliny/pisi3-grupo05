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

# Carregando os datasets necessários
df_negocios = pd.read_parquet('yelp_academic_dataset_business_cleaned.parquet')
df_tip = pd.read_parquet('yelp_academic_dataset_tip_cleaned.parquet')

# Título principal do aplicativo Streamlit
st.title('Análise de Negócios do Yelp')

# Seção 1: Vizualização das categorias mais frequentes
st.subheader('Top 10 Categorias Mais Frequentes')

# Calculando a contagem das 10 categorias mais frequentes
category_counts = df_negocios['categories'].value_counts().head(10)

# Criando um gráfico de barras para exibir as categorias mais frequentes
fig, ax = plt.subplots(figsize=(14, 8))
sns.barplot(x=category_counts.values, y=category_counts.index, ax=ax)
ax.set_title('Top 10 Categorias Mais Frequentes')
ax.set_xlabel('Contagem')
ax.set_ylabel('Categorias')
st.pyplot(fig)

st.subheader('Categorias mais populares no Yelp')
st.write('As categorias mais frequentes indicam que os negócios mais comuns no Yelp são restaurantes e cafés, com forte presença de empresas voltadas para alimentação.')

# Seção 2: Distribuição da faixa de preço
st.subheader('Distribuição da Faixa de Preço')

# Criando um histograma para visualizar a distribuição da faixa de preço
fig, ax = plt.subplots(figsize=(10, 6))
bins = [0.5, 1.5, 2.5, 3.5, 4.5]  # Definindo os limites para cada faixa de preço
sns.histplot(df_negocios['RestaurantsPriceRange2'], bins=bins, kde=False, color='skyblue', edgecolor='black', stat='frequency', ax=ax)
ax.set_xticks([1, 2, 3, 4])  # Ajustando os ticks do eixo X
ax.set_title('Distribuição da Faixa de Preço', fontsize=14)
ax.set_xlabel('Faixa de Preço', fontsize=12)
ax.set_ylabel('Frequência', fontsize=12)

# Adicionando rótulos com a contagem de cada barra
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), 
                textcoords='offset points')
st.pyplot(fig)

st.subheader('Faixa de preço mais procurada')
st.write('A maioria dos negócios está concentrada nas faixas de preço mais baixas e médias, sugerindo que os consumidores do Yelp tendem a buscar opções acessíveis, mas com boa variedade.')

# Seção 3: Distribuição das estrelas
st.subheader('Distribuição das Estrelas')

# Criando um histograma para a distribuição das estrelas
fig, ax = plt.subplots(figsize=(10, 6))
hist_data = sns.histplot(df_negocios['stars'], bins=10, kde=False, ax=ax, color='skyblue', edgecolor='black')
hist_data.set_ylabel('Frequência')

# Ajustando os dados para a suavização da curva
x = [patch.get_x() + patch.get_width() / 2 for patch in hist_data.patches]
y = [patch.get_height() for patch in hist_data.patches]
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(np.array(x).reshape(-1, 1))
poly_reg_model = LinearRegression()
poly_reg_model.fit(x_poly, y)
x_full = np.linspace(min(x), max(x), 200)
x_full_poly = poly.fit_transform(x_full.reshape(-1, 1))
y_smooth = poly_reg_model.predict(x_full_poly)

# Adicionando uma curva suave ao histograma
ax.plot(x_full, y_smooth, color='red', linewidth=2, label='Curva Suave')
ax.set_title('Distribuição das Estrelas')
ax.set_xlabel('Estrelas')
ax.set_ylabel('Frequência')
ax.legend()
st.pyplot(fig)


st.subheader('Avaliações mais comuns no Yelp')
st.write('A maioria das avaliações estão concentradas em 4 e 5 estrelas, indicando que os consumidores tendem a ser bastante positivos em suas avaliações no Yelp.')

# Seção 4: Análise da relação entre a faixa de preço e estrelas
st.subheader('Relação entre Faixa de Preço e Número de Estrelas ')

# Criando um boxplot para mostrar a relação
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='RestaurantsPriceRange2', y='stars', data=df_negocios, ax=ax)
ax.set_title('Relação entre Faixa de Preço e Número de Estrelas ')
ax.set_xlabel('Faixa de Preço')
ax.set_ylabel('Estrelas')
st.pyplot(fig)

st.subheader('Faixa de preço e sua relação com a qualidade')
st.write('Negócios com preços mais baixos têm uma maior variação nas avaliações, enquanto os de preços mais altos possuem avaliações mais consistentes, sugerindo uma expectativa de qualidade mais alta.')

# Seção 5: Removendo outliers para análise mais precisa
st.subheader('Relação entre Faixa de Preço e Número de Estrelas (Sem Outliers)')

# Filtrando os dados para remover outliers usando o método do IQR
Q1 = df_negocios['stars'].quantile(0.25)
Q3 = df_negocios['stars'].quantile(0.75)
IQR = Q3 - Q1
df_negocios_filtered = df_negocios[~((df_negocios['stars'] < (Q1 - 1.5 * IQR)) | (df_negocios['stars'] > (Q3 + 1.5 * IQR)))]

# Criando outro boxplot com os dados filtrados
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='RestaurantsPriceRange2', y='stars', data=df_negocios_filtered, ax=ax)
ax.set_title('Relação entre Faixa de Preço e Número de Estrelas (Sem Outliers)')
ax.set_xlabel('Faixa de Preço')
ax.set_ylabel('Estrelas')
st.pyplot(fig)

st.subheader('Efeito dos outliers nas análises')
st.write('Após a remoção dos outliers, a relação entre faixa de preço e estrelas se torna mais clara, mostrando que faixas de preço mais altas estão mais associadas a avaliações mais positivas.')

# Seção 6: Análise temporal das avaliações
st.subheader('Tendência da Média de Avaliações ao Longo do Tempo')

# Criando uma série temporal para mostrar a tendência
df_tip['year_month'] = df_tip['date'].dt.to_period('M')
stars_over_time = df_tip.merge(df_negocios[['business_id', 'stars']], on='business_id').groupby('year_month')['stars'].mean()

# Plotando a tendência
fig, ax = plt.subplots(figsize=(14, 8))
stars_over_time.plot(ax=ax)
ax.set_title('Tendência da Média de Avaliações ao Longo do Tempo')
ax.set_xlabel('Ano e Mês')
ax.set_ylabel('Média de Estrelas')
st.pyplot(fig)

st.subheader('Tendência temporal das avaliações')
st.write('A média de avaliações tende a melhorar ao longo do tempo, indicando que os negócios estão se esforçando para atender às expectativas dos consumidores.')

# Seção 7: Gráfico de dispersão
st.subheader('Quantidade de Comentários x Avaliações')

# Para os pontos não ficarem sobrepostos
jitter_strength = 0.1
df_tip['stars_jittered'] = df_tip['stars'] + np.random.uniform(-jitter_strength, jitter_strength, size=df_tip.shape[0])
df_tip['total_comments_jittered'] = df_tip['total_comments'] + np.random.uniform(-jitter_strength, jitter_strength, size=df_tip.shape[0])

# Comentários x Avaliações
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df_tip['stars_jittered'], df_tip['total_comments_jittered'], alpha=0.5, color='blue')
ax.set_title('Quantidade de Comentários x Avaliações', fontsize=16)
ax.set_xlabel('Avaliações', fontsize=12)
ax.set_ylabel('Comentários', fontsize=12)
ax.set_ylim(0, 3000)
ax.grid(False)
st.pyplot(fig)

st.subheader('Análise da relação entre comentários e avaliações')
st.write('Este gráfico mostra a relação entre o número de avaliações e a quantidade total de comentários para cada estabelecimento no Yelp.')

# Seção 8: Mapa de calor para popularidade regional
st.subheader('Popularidade Regional: Mapa de Calor')

# Criando um mapa com os dados de localização
mapa = folium.Map(location=[df_negocios['latitude'].mean(), df_negocios['longitude'].mean()], zoom_start=10)
heat_data = [[row['latitude'], row['longitude'], row['stars']] for index, row in df_negocios.iterrows()]
HeatMap(heat_data).add_to(mapa)
st_folium(mapa, width=700, height=500)
st.write('O mapa de calor revela áreas com alta concentração de negócios bem avaliados, ajudando a identificar regiões populares e com boa satisfação dos clientes.')

# Seção 9: Análise das 10 cidades com mais estabelecimentos bem avaliados
st.subheader('Top 10 Cidades com Mais Estabelecimentos Bem Avaliados')

# Filtrar estabelecimentos bem avaliados (por exemplo, com estrelas >= 4)
estabelecimentos_bem_avaliados = df_negocios[df_negocios['stars'] >= 4]

# Agrupar por cidade e calcular a média de avaliação e o total de estabelecimentos
cidade_bem_avaliada = estabelecimentos_bem_avaliados.groupby('city').agg(
    media_avaliacao=('stars', 'mean'),
    total_estabelecimentos=('business_id', 'count')
).sort_values(by='total_estabelecimentos', ascending=False)

# Exibir as top 10 cidades com mais estabelecimentos bem avaliados
st.write("Top 10 cidades com mais estabelecimentos bem avaliados:")
st.dataframe(cidade_bem_avaliada.head(10))

# Insight sobre as cidades com mais estabelecimentos bem avaliados
st.write('Esses dados podem ser usados para responder perguntas como: quais são as características comuns entre os estabelecimentos bem avaliados nessas cidades? '
         'Como a localização geográfica influencia a qualidade percebida dos serviços? '
         'Além de que podem ajudar a identificar padrões e tendências que podem ser aplicados para melhorar a qualidade dos serviços em outras regiões.')

st.subheader('Mapa de calor da popularidade')

# Seção 10: Estabelecimentos bem avaliados nas 10 cidades mais populares
st.subheader('Estabelecimentos Bem Avaliados nas 10 Cidades Mais Populares')

# Filtrar estabelecimentos bem avaliados (média de estrelas > 4)
estabelecimentos_bem_avaliados = df_negocios[df_negocios['stars'] > 4]

# Listagem dos estabelecimentos bem avaliados nas 10 cidades mais populares
top_10_cidades = cidade_bem_avaliada.head(10).index

# Filtrando os negócios nas 10 cidades mais populares
estabelecimentos_populares = estabelecimentos_bem_avaliados[estabelecimentos_bem_avaliados['city'].isin(top_10_cidades)]

# Exibindo detalhes dos estabelecimentos
st.write("Estabelecimentos bem avaliados nas 10 cidades mais populares:")
st.dataframe(estabelecimentos_populares[['name', 'city', 'categories']].head(10))

# Insight sobre os estabelecimentos bem avaliados nas cidades mais populares
st.write('Analisar os estabelecimentos bem avaliados nas cidades mais populares pode fornecer insights sobre as características que contribuem para avaliações positivas. '
         'Quais são as categorias de negócios mais comuns entre os estabelecimentos bem avaliados? '
         'Quais práticas ou características específicas estão associadas a avaliações positivas? '
         )
