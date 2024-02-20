import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
import threading
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import sys

# LEITURA E TRATAMENTO DOS DADOS
df = pd.read_csv('dataset - projeto.csv', sep=',', encoding='latin1')
df.duplicated().sum()

# DECLARACAO DA CLASS AMBIENTE
class Ambiente:
    def __init__(self, tamanho, probabilidade_livre, probabilidade_bomba, probabilidade_tesouro):
        self.tamanho = tamanho
        self.grid = self.inicializar_ambiente(probabilidade_livre, probabilidade_bomba, probabilidade_tesouro)
        self.agentes = []
        self.agentes_destruidos = []
        self.marcador = []
        # REASTREAMENTO DOS MODELOS QUE EXPLORAM O AMBIENTE
        self.celulas_visitadas_por_modelo = {'modelo_kneighborsclassifier': 0, 'modelo_decisiontreeclassifier': 0, 'modelo_randomforestclassifier': 0}
        self.total_tesouros = sum(linha.count('T') for linha in self.grid)
        self.simulacao_ativa = True
        # Configurações do Tkinter
        self.CELL_SIZE = 60
        self.WIDTH, self.HEIGHT = tamanho * self.CELL_SIZE, tamanho * self.CELL_SIZE
        # Inicialização da tela
        self.root = tk.Tk()
        self.root.title("Exploração De Ambiente Desconhecido")
        # containter principar
        self.container = tk.Frame(self.root)
        self.container.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.root, width=self.WIDTH, height=self.HEIGHT)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Widget de texto para exibir informações
        self.info_text = tk.Text(self.root, width=63, height=18)
        self.info_text.pack()
        # Separador vertical
        ttk.Separator(self.container, orient=tk.HORIZONTAL).pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # Botões
        self.btn1 = tk.Button(self.root, text="Modelo knn",command= Modelo_knn.run )
        self.btn1.pack(side=tk.TOP, padx=5, pady=5)
        self.btn3 = tk.Button(self.root, text="Arvore de Decisao",command= Modelo_Dtree.run )
        self.btn3.pack(side=tk.TOP, padx=5, pady=5)
        self.btn2 = tk.Button(self.root, text="Modelo Random Forest", command = Modelo_RForest.run )
        self.btn2.pack(side=tk.TOP, padx=5, pady=5)
        self.btn7 = tk.Button(self.root, text="Algoritmos Combinados", command = Main.run)
        self.btn7.pack(side=tk.TOP, padx=5, pady=5)
        self.btn5 = tk.Button(self.root, text="Estatistica", command= self.mensagem)
        self.btn5.pack(side=tk.TOP, padx=5, pady=5)
        self.btn4 = tk.Button(self.root, text=" Histograma", command= self.mostrar_histograma_celulas_visitadas)
        self.btn4.pack(side=tk.TOP, padx=5, pady=5)
        self.btn4 = tk.Button(self.root, text="Analisar Graficamente", command= self.mostrar_grafico_celulas_visitadas)
        self.btn4.pack(side=tk.TOP, padx=5, pady=5)

        # Redirecionamento da saída de impressão
        sys.stdout = self
        self.root.protocol("WM_DELETE_WINDOW", self.encerrar_simulacao)
        self.start_time = 0
        self.end =0
        self.tempos_execucao = []
        
    def adicionar_informacao(self, info):
        self.info_text.insert(tk.END, info)
        self.info_text.see(tk.END)

    def write(self, text):
        self.adicionar_informacao(text)

    def inicializar_ambiente(self, probabilidade_livre, probabilidade_bomba, probabilidade_tesouro):
        ambiente = [['L' if random.random() < probabilidade_livre
                     else 'B' if random.random() < probabilidade_bomba
                     else 'T' if random.random() < probabilidade_tesouro
                     else 'L' for _ in range(self.tamanho)] for _ in range(self.tamanho)]
        return ambiente

    def adicionar_agente(self, agentes):
        for agente in agentes:
            agente.tamanho = self.tamanho
            self.agentes.append(agente)

    def desenhar_ambiente(self):
        self.canvas.delete('all')
        for i, linha in enumerate(self.grid):
            for j, celula in enumerate(linha):
                self.marcador= self.agentes + self.agentes_destruidos
                valor = celula
                visitada_por_agente = any(((i, j) in agente.visitadas) for agente in self.marcador)       
                cor_fundo = "yellow" if visitada_por_agente else "white"
                cor_agente = "red" if any(agente.agent_pos == (i, j) for agente in self.agentes) else "black"
                self.canvas.create_rectangle(j * self.CELL_SIZE, i * self.CELL_SIZE,
                                            (j + 1) * self.CELL_SIZE, (i + 1) * self.CELL_SIZE,
                                            fill=cor_fundo, outline="gray")
                self.canvas.create_text((j + 0.5) * self.CELL_SIZE, (i + 0.5) * self.CELL_SIZE,
                                        text=valor, fill="black")
                for agente in self.agentes:
                    if agente.agent_pos == (i, j):
                        self.canvas.create_oval((j + 0.25) * self.CELL_SIZE, (i + 0.25) * self.CELL_SIZE,
                                                (j + 0.75) * self.CELL_SIZE, (i + 0.75) * self.CELL_SIZE,
                                                fill=cor_agente, outline="black")  
            
        self.root.update()
        self.marcador = []

    def mover_agente(self, agente, acao, ambiente):
        agente.mover(acao, self)
        self.desenhar_ambiente()
        for agente_ativo in ambiente.agentes:
            if agente == agente_ativo:
                agente.informacoes_compartilhadas(ambiente)
                break
        
    def destruir_agente(self, agente):
        if agente not in self.agentes_destruidos:
            self.agentes_destruidos.append(agente)
            print(f"Agente {agente.nome} foi destruído.")
            # Remover o agente da lista de agentes
            self.agentes.remove(agente)
        
        self.desenhar_ambiente()
                
    def encerrar_simulacao(self):
        self.simulacao_ativa = False
        self.btn_encerrar = tk.Button(self.root, text="Encerrar Simulação", command=self.destruir_janela)
        self.btn_encerrar.pack(side=tk.TOP, padx=5, pady=5)

    def destruir_janela(self):
        self.root.destroy()

    def mensagem(self):
        # Limpar o texto anterior
        self.info_text.delete("1.0", tk.END)
        print('============= ANALISE ESTATISTICO DA SIMULACAO ===============', "\n")
        tempo_total = sum(self.tempos_execucao)
        print(f"Tempo total de execução para todos os agentes: {tempo_total:.2f} min")
        # Total de bombas, tesouros e espaços livres no ambiente
        contagem_bombas = sum(linha.count('B') for linha in self.grid)
        contagem_tesouros = sum(linha.count('T') for linha in self.grid)
        contagem_livres = sum(linha.count('L') for linha in self.grid)
        print(f"Total de bombas no ambiente: {contagem_bombas}")
        print(f"Total de tesouros no ambiente: {contagem_tesouros}")
        print(f"Total de espaços livres no ambiente: {contagem_livres}")
        # Total de tesouros coletados por todos os agentes
        total_tesouros_agent_ativo = sum(agente.tesouros_encontrados for agente in self.agentes)
        total_tesouros_agent_inativo = sum(agente.tesouros_encontrados for agente in self.agentes_destruidos)
        total_tesouros_coletados = total_tesouros_agent_ativo + total_tesouros_agent_inativo
        print(f"Total de tesouros coletados por todos os agentes: {total_tesouros_coletados}")
       # Agente que mais coletou tesouros
        agentes_tesouros_ativo = {agente.nome: agente.tesouros_encontrados for agente in self.agentes}
        agentes_tesouros_inativo = {agente.nome: agente.tesouros_encontrados for agente in self.agentes_destruidos}
        # Combine os dicionários de tesouros coletados pelos agentes ativos e inativos
        todos_agentes_tesouros = {**agentes_tesouros_ativo, **agentes_tesouros_inativo}
        # Encontre o agente que mais coletou tesouros em todos os agentes
        if todos_agentes_tesouros:
            agente_mais_tesouros = max(todos_agentes_tesouros, key=todos_agentes_tesouros.get)
            print(f"Agente que mais coletou tesouros: {agente_mais_tesouros}")
        else:
            print("Nenhum agente coletou tesouros.")
               
    def mostrar_grafico_celulas_visitadas(self):
        modelos = list(self.celulas_visitadas_por_modelo.keys())
        celulas_visitadas = list(self.celulas_visitadas_por_modelo.values())
        # Definindo cores para as barras
        cores = ['blue', 'green', 'orange', 'red', 'purple', 'yellow']
        plt.figure(figsize=(10, 6))
        # Plotando as barras com cores diferentes
        plt.bar(modelos, celulas_visitadas, color=cores)
        plt.xlabel('Modelos')
        plt.ylabel('Número de Células Visitadas')
        plt.title('Número de Células Visitadas por Modelo')
        # Adicionando legenda para cores
        plt.legend(labels=modelos)
        # Adicionando grade
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # Rotacionando os rótulos do eixo x para melhor legibilidade
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def mostrar_histograma_celulas_visitadas(self):
        # modelos = list(self.celulas_visitadas_por_modelo.keys())
        # celulas_visitadas = list(self.celulas_visitadas_por_modelo.values())
        celulas_visitadas = list(self.celulas_visitadas_por_modelo.values())
        plt.figure(figsize=(10, 6))
        plt.hist(celulas_visitadas, bins=10, color='blue', edgecolor='black')
        plt.xlabel('Número de Células Visitadas')
        plt.ylabel('Frequência')
        plt.title('Histograma do Número de Células Visitadas')
        plt.grid(True)
        plt.show()


# DECLARACAO DA CLASS AGENTE
class Agente:
    def __init__(self, nome, modelos):
        self.ultima_acao = None
        self.agent_pos = None
        self.tamanho = None
        self.nome = nome
        self.forca = 1
        self.tesouros_encontrados = 0
        self.tesouros_coletados = set() 
        self.visitadas = set()
        self.modelos = modelos  # três modelos
        self.modelo_votacao = False
        # Adicionando contador para evitar loops
        self.contador_acao_repetida = 0
        self.ultima_posicao = None
        self.registro_movimentos = [] 
        self.registro_tamanho = 30
        self.bandeira_encontrada = False
        self.agente_sombra = AgenteSombra()

    def iniciar_agente(self, tamanho):
        self.tamanho = tamanho
        self.agent_pos = (random.randint(0, tamanho - 1), random.randint(0, tamanho - 1))
        self.visitadas.add(self.agent_pos)

    def explorar_inteligentemente(self, ambiente):
        if self in ambiente.agentes_destruidos:
            return None
        
        movimento = ['B', 'C', 'D', 'E']
        observacoes = self.obter_estado_agente(ambiente)
        entrada = np.array([observacoes])
        previsoes = []

        # Lidar com diferentes tipos de modelos
        if hasattr(self.modelos, '__iter__'):
            # Iterar sobre os modelos se for iterável
            for modelo in self.modelos:
                resultado = modelo.predict(entrada)
                if hasattr(resultado, '__iter__'):
                    previsoes.append(resultado[0])
                else:
                    previsoes.append(resultado)
        else:
            # Se apenas um modelo for fornecido
            resultado = self.modelos.predict(entrada)
            if hasattr(resultado, '__iter__'):
                previsoes.append(resultado[0])
            else:
                previsoes.append(resultado)

        # Escolher a ação com base na estratégia selecionada, verifica se todos as posicoes sao diferente       
        if all(previsoes[i] != previsoes[j] for i in range(len(previsoes)) for j in range(i + 1, len(previsoes))):
            predicted_action = self.ponderar(previsoes)
            mover = movimento[int(predicted_action)]
            return mover
        else:
            lets_vote  = self.votar(previsoes)
            if lets_vote is not None: 
                mover = movimento[int(lets_vote)]
                return mover
            

    def votar(self, previsoes):
        votos = Counter(previsoes)
        previsao_mais_comum = votos.most_common(1)[0][0] 
        if previsao_mais_comum >= 0:
            return previsao_mais_comum
        else:
            return None
    
    def ponderar(self, previsoes):
        try:
            num_modelos = len(previsoes)
            peso_por_modelo = 1 / num_modelos
            pesos = [peso_por_modelo] * num_modelos
            pontuacoes_ponderadas = [np.dot(previsao, pesos) for previsao in previsoes]
            indice_modelo = np.argmax(pontuacoes_ponderadas)
            # Garanta que o índice esteja no intervalo de 0 a 3
            indice_modelo = max(0, min(indice_modelo, 3))
            return indice_modelo
        
        except TypeError:
            print("Erro: previsões inválidas. ", indice_modelo)
            return 0 

    def obter_estado_agente(self, ambiente):
        x, y = self.agent_pos
        proximidades = [
            ambiente.grid[x][y+1] if 0 <= y+1 < ambiente.tamanho else 'V',
            ambiente.grid[x][y-1] if 0 <= y-1 < ambiente.tamanho else 'V',
            ambiente.grid[x-1][y] if 0 <= x-1 < ambiente.tamanho else 'V',
            ambiente.grid[x+1][y] if 0 <= x+1 < ambiente.tamanho else 'V'
        ]
        return [1 if p == 'L' else 2 if p == 'T' else 0 if p == 'B' else 3 for p in proximidades]
    
    def mover(self, acao, ambiente):
        if acao == None:
            return
        
        # Chamando o AgenteSombra para obter o melhor trajeto
        melhor_trajeto = self.agente_sombra.encontrar_melhor_trajeto(self.agent_pos, ambiente)
        if melhor_trajeto == -1:
            mover_para = acao
        else:
            mover_para = melhor_trajeto
    
        x, y = self.agent_pos
        nova_posicao = self.calcular_nova_posicao(mover_para, x, y)
        self.atualizar_registro_movimentos(nova_posicao)
        if self.detectar_loop_movimento():
            mover_para = self.escolher_acao_aleatoria(mover_para)
            nova_posicao = self.calcular_nova_posicao(mover_para, x, y)

        # Verificar se o movimento é válido
        if self.eh_movimento_valido(nova_posicao, ambiente):
            if nova_posicao not in self.visitadas:
                self.visitadas.add(nova_posicao)    
                modelo_agente = self.modelos.__class__.__name__.lower()
                chave_modelo = f'modelo_{modelo_agente}'
                if chave_modelo in ambiente.celulas_visitadas_por_modelo:
                    ambiente.celulas_visitadas_por_modelo[chave_modelo] += 1

            self.agent_pos = nova_posicao
            self.atualizar_estado(ambiente)
            print('posicao ===> ', nova_posicao)
        else:
            mover_para = self.escolher_acao_aleatoria(mover_para)
            nova_posicao = self.calcular_nova_posicao(mover_para, x, y)

    def calcular_nova_posicao(self, acao, x, y):
        tamanho = self.tamanho
        if acao == 'C':
            return (x-1, y) if x-1 >= 0 else (x+1, y)  
        elif acao == 'D':
            return (x, y+1) if y+1 < tamanho else (x, y-1) 
        elif acao == 'B':
            return (x+1, y) if x+1 < tamanho else (x-1, y)  
        elif acao == 'E':
            return (x, y-1) if y-1 >= 0 else (x, y+1) 

    def atualizar_registro_movimentos(self, nova_posicao):
        self.registro_movimentos.append(nova_posicao)
        if len(self.registro_movimentos) > self.registro_tamanho:
            self.registro_movimentos.pop(0)
    
    def detectar_loop_movimento(self):
        if len(self.registro_movimentos) < self.registro_tamanho:
            return False
        
        repeticoes_minimas = 3
        posicoes_repetidas = []
        
        for i in range(len(self.registro_movimentos) - self.registro_tamanho + 1):
            sequencia_atual = self.registro_movimentos[i:i+self.registro_tamanho]
            sequencia_oposta = [(y, x) for x, y in sequencia_atual[::-1]]  # Inverte as coordenadas
            contagem_atual = Counter(sequencia_atual)
            contagem_oposta = Counter(sequencia_oposta)
            posicoes_repetidas.extend([posicao for posicao, contagem in contagem_atual.items() if contagem >= repeticoes_minimas])
            posicoes_repetidas.extend([posicao for posicao, contagem in contagem_oposta.items() if contagem >= repeticoes_minimas])
        
        posicoes_repetidas = list(set(posicoes_repetidas))  # Remove duplicatas
        if posicoes_repetidas:
            print("Loop detectado! Posições repetidas:", posicoes_repetidas)

        return len(posicoes_repetidas) > 1

    def eh_movimento_valido(self, nova_posicao, ambiente):
        return 0 <= nova_posicao[0] < self.tamanho and 0 <= nova_posicao[1] < self.tamanho

    def atualizar_estado(self, ambiente):
        x, y = self.agent_pos
        celula_atual = ambiente.grid[x][y]
        
        if celula_atual == 'T':
            if (x, y) not in self.tesouros_coletados:  
                self.tesouros_encontrados += 1  
                self.tesouros_coletados.add((x, y))  
                self.forca = max(1, self.forca + 1)  
            
            if self.tesouros_encontrados > ambiente.total_tesouros // 2:
                print(f"Agente {self.nome} encontrou 50% dos tesouros. Parando.")
                ambiente.encerrar_simulacao()
            
            if len(ambiente.agentes)==0:
                print("Todos os agentes morreram. Simulação encerrada sem sucesso.")
                ambiente.encerrar_simulacao()

        elif celula_atual == 'B':
            self.forca -= 1
            if self.forca == 0:
                self.visitadas.add(self.agent_pos)
                self.informacoes_compartilhadas(ambiente)
                ambiente.destruir_agente(self)
            
            if len(ambiente.agentes)==0:
                print("Todos os agentes morreram. Simulação encerrada sem sucesso.")
                ambiente.encerrar_simulacao()

    def escolher_acao_aleatoria(self, acao_atual):
        movimentos = ['C', 'D', 'B', 'E']
        movimentos.remove(acao_atual)
        return random.choice(movimentos)
        
    def informacoes_compartilhadas(self, ambiente):
        x, y = self.agent_pos
        vizinhos = [
            (x, y+1) if 0 <= y+1 < ambiente.tamanho else None,
            (x, y-1) if 0 <= y-1 < ambiente.tamanho else None,
            (x-1, y) if 0 <= x-1 < ambiente.tamanho else None,
            (x+1, y) if 0 <= x+1 < ambiente.tamanho else None
        ]
        
        # Lista de prioridades para escolha de movimento
        prioridades = {
            'T': 2,  
            'L': 1,  
            'B': 0,  
            'V': -1   
        }
        
        for vizinho in vizinhos:
            if vizinho is not None:
                i, j = vizinho
                visitado_por_agente = any(vizinho in agente.visitadas for agente in ambiente.agentes)
                celula_vizinha = '?' 
                if visitado_por_agente:
                    info_vizinho = self.obter_informacoes_vizinho(vizinho, ambiente)
                    celula_vizinha = info_vizinho['celula']
                    print(f'Tem um : {celula_vizinha} na posição: {vizinho}, partilhado por:{self.nome} \n')
                
    def obter_informacoes_vizinho(self, vizinho, ambiente):
        x, y = vizinho
        celula = ambiente.grid[x][y]
        return {'celula': celula, }


class AgenteSombra:
    def __init__(self):
        pass

    def encontrar_melhor_trajeto(self, posicao_agente, ambiente):
        melhor_trajeto = None
        melhor_pontuacao = float('-inf')
        todas_pontuacoes_negativas = True
        # Define as direções e seus incrementos de posição correspondentes
        direcoes = {
            'D': (0, 1),
            'E': (0, -1),
            'C': (-1, 0),
            'B': (1, 0)
        }
        
        for direcao, incremento in direcoes.items():
            pontuacao_direcao = self.pontuar_trajeto(posicao_agente, incremento, ambiente)
            if pontuacao_direcao > melhor_pontuacao:
                melhor_pontuacao = pontuacao_direcao
                melhor_trajeto = direcao
                todas_pontuacoes_negativas = False
            
            # Se a pontuação da direção atual for positiva ou zero, atualizamos o flag
            if pontuacao_direcao > 0:
                todas_pontuacoes_negativas = False
        # Verifica se todas as pontuações foram negativas
        if todas_pontuacoes_negativas:
            return -1
        return melhor_trajeto

    
    def pontuar_trajeto(self, posicao_agente, incremento, ambiente):
        pontuacao = 0
        x, y = posicao_agente
        # Explora os próximos três passos na direção especificada
        for _ in range(1, 4):
            x += incremento[0]
            y += incremento[1]
            # Verifica se a posição está dentro dos limites do ambiente
            if 0 <= x < ambiente.tamanho and 0 <= y < ambiente.tamanho:
                celula = ambiente.grid[x][y]
                # Atribui pontuação com base no conteúdo da célula
                if celula == 'B': 
                    pontuacao -= 10
                elif celula == 'T': 
                    pontuacao += 5
                else: 
                    pontuacao += 1
            else:
                # Penalização por sair dos limites
                pontuacao -= 5  
        return pontuacao

from sklearn.preprocessing import LabelEncoder
df = pd.DataFrame(df)
# CONVERTENDO DADOS CATEGORICOS EM NUMERICO.
label_encoder = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].apply(label_encoder.fit_transform)

X = df.drop('4', axis=1)
y = df['4']

# CRIACAO DE VARIACOES DOS DADAS
def gerar_variacoes_observacoes(observacoes):
    variacoes = []
    variacoes_acao = []

    for i in range(len(observacoes)):
        try:
            variacao_obs = observacoes.copy()  # Cria uma cópia de observacoes
            variacao_obs[i] = abs(1 - variacao_obs[i])  # Garantir que o valor seja positivo
            # Garantir que a variável de observação esteja no intervalo [0, 3]
            variacao_obs[i] = max(0, min(variacao_obs[i], 3))
            variacoes.append(variacao_obs)
            variacao_acao =  random.randint(0, 3)  # Variação simples da ação
            variacoes_acao.append(variacao_acao)
        except IndexError:
            pass

    return variacoes, variacoes_acao

def augmentar_dados(X, y):
    X_aumentado = []
    y_aumentado = []

    for observacoes, acao in zip(X, y):
        try:
            variacoes_observacoes, variacoes_acao = gerar_variacoes_observacoes(observacoes)
            conjunto_variacoes = set(map(tuple, variacoes_observacoes))
            X_aumentado.extend( conjunto_variacoes)
            y_aumentado.extend(variacoes_acao)
        except IndexError:
            pass

    # Escrever as variações no arquivo após o loop for
    with open('variacoes.txt', 'w') as arquivo:
        for observacao, acao in zip(X_aumentado, y_aumentado):
            observacao_str = ' '.join(map(str, observacao))
            linha = f"{observacao_str}=>{acao}\n"
            arquivo.write(linha)
            
    return X_aumentado, y_aumentado

import threading
# ====================== CLASS MAIN =====================
class Main:
    @staticmethod
    def run():
        
        num_agentes = 5
        bomba = 0.4
        tesouro = 0.3


        X_aumentado, y_aumentado = augmentar_dados(X.values , y)
        X_train, X_test, y_train, y_test = train_test_split(X_aumentado, y_aumentado , test_size=0.2, random_state=42)
        # Treinando o modelo de árvore de decisão
        modelo_arvore_decisao = DecisionTreeClassifier(criterion='entropy')
        modelo_arvore_decisao.fit(X_train, y_train)
        # Treinando o modelo KNN
        modelo_knn = KNeighborsClassifier(n_neighbors=3)
        modelo_knn.fit(X_train, y_train)
        # Treinando o modelo Random Forest
        modelo_random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo_random_forest.fit(X_train, y_train)

        modelos_disponiveis = [modelo_arvore_decisao, modelo_knn, modelo_random_forest]

        agentes = [Agente(f"agente{i+1}", random.choice(modelos_disponiveis)) for i in range(num_agentes)]
        ambiente = Ambiente(tamanho=10, probabilidade_livre=1-bomba-tesouro, probabilidade_bomba=bomba, probabilidade_tesouro=tesouro)
        ambiente.adicionar_agente(agentes) 
    
        for agente in agentes:
            agente.iniciar_agente(tamanho=10)

        while ambiente.simulacao_ativa:
            for agente in agentes:
                ambiente.start_time = time.time()  
                try:
                    acao_escolhida = agente.explorar_inteligentemente(ambiente)
                except TypeError:
                    pass
                ambiente.mover_agente(agente, acao_escolhida, ambiente)
                time.sleep(0.2)
                ambiente.end_time = time.time()  
                # Atualize os tempos_execucao
                ambiente.tempo_execucao_agente = (ambiente.end_time - ambiente.start_time)/60
                ambiente.tempos_execucao.append(ambiente.tempo_execucao_agente)


        # Exibir o percurso de cada agente após a simulação
        for agente in agentes:
            print(f"O agente {agente.nome} encontrou {agente.tesouros_encontrados} tesouros.")

def main():
    if __name__ == "__main__":
        threading.Thread(target=Main.run).start()

#============================== MODELO KNN ==================================

class Modelo_knn:
    @staticmethod
    def run():
        num_agentes = 5
        bomba = 0.4
        tesouro = 0.3

        X_aumentado, y_aumentado = augmentar_dados(X.values , y)
        X_train, X_test, y_train, y_test = train_test_split(X_aumentado, y_aumentado , test_size=0.2, random_state=42)
    
        # Treinando o modelo KNN
        modelo_knn = KNeighborsClassifier(n_neighbors=3)
        modelo_knn.fit(X_train, y_train)
      
        agentes = [Agente(f"agente{i+1}", modelo_knn) for i in range(num_agentes)]
        ambiente = Ambiente(tamanho=10, probabilidade_livre=1-bomba-tesouro, probabilidade_bomba=bomba, probabilidade_tesouro=tesouro)
        ambiente.adicionar_agente(agentes) 
    
        for agente in agentes:
            agente.iniciar_agente(tamanho=10)

        while ambiente.simulacao_ativa:
            for agente in agentes:
                ambiente.start_time = time.time()  
                try:
                    acao_escolhida = agente.explorar_inteligentemente(ambiente)
                except TypeError:
                    pass
                ambiente.mover_agente(agente, acao_escolhida, ambiente)
                time.sleep(0.2)
                ambiente.end_time = time.time()  
                # Atualize os tempos_execucao
                ambiente.tempo_execucao_agente = (ambiente.end_time - ambiente.start_time)/60
                ambiente.tempos_execucao.append(ambiente.tempo_execucao_agente)
        # Exibir o percurso de cada agente após a simulação
        for agente in agentes:
            print(f"O agente {agente.nome} encontrou {agente.tesouros_encontrados} tesouros.")

def run_knn():
    if __name__ == "__main__":
        threading.Thread(target= Modelo_knn.run).start()    

# # # ====================== MODELO ARVORE DE DECISAO ======================
class Modelo_Dtree:
    @staticmethod
    def run():
        num_agentes = 5
        bomba = 0.4
        tesouro = 0.3

        X_aumentado, y_aumentado = augmentar_dados(X.values , y)
        X_train, X_test, y_train, y_test = train_test_split(X_aumentado, y_aumentado , test_size=0.2, random_state=42)
        # Treinando o modelo de árvore de decisão
        modelo_arvore_decisao = DecisionTreeClassifier(criterion='entropy')
        modelo_arvore_decisao.fit(X_train, y_train)

        agentes = [Agente(f"agente{i+1}", modelo_arvore_decisao) for i in range(num_agentes)]
        ambiente = Ambiente(tamanho=10, probabilidade_livre=1-bomba-tesouro, probabilidade_bomba=bomba, probabilidade_tesouro=tesouro)
        ambiente.adicionar_agente(agentes) 

        for agente in agentes:
            agente.iniciar_agente(tamanho=10)

        while ambiente.simulacao_ativa:
            for agente in agentes:
                ambiente.start_time = time.time()  
                acao_escolhida = agente.explorar_inteligentemente(ambiente)
                ambiente.mover_agente(agente, acao_escolhida,ambiente)
                time.sleep(0.2)
                ambiente.end_time = time.time()  
                # Atualize os tempos_execucao
                ambiente.tempo_execucao_agente = (ambiente.end_time - ambiente.start_time)/60
                ambiente.tempos_execucao.append(ambiente.tempo_execucao_agente)

def run_Dtree():
    if __name__ == "__main__":
        threading.Thread(target=Modelo_Dtree.run).start()

# # ====================== MODELO RANDOM FOREST ======================
class Modelo_RForest:
    @staticmethod
    def run():
        num_agentes = 5
        bomba = 0.4
        tesouro = 0.3

        X_aumentado, y_aumentado = augmentar_dados(X.values , y)
        X_train, X_test, y_train, y_test = train_test_split(X_aumentado, y_aumentado , test_size=0.2, random_state=42)
        # Treinando o modelo de árvore de decisão
        modelo_random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo_random_forest.fit(X_train, y_train)

        agentes = [Agente(f"agente{i+1}", modelo_random_forest) for i in range(num_agentes)]
        ambiente = Ambiente(tamanho=10, probabilidade_livre=1-bomba-tesouro, probabilidade_bomba=bomba, probabilidade_tesouro=tesouro)
        ambiente.adicionar_agente(agentes) 

        for agente in agentes:
            agente.iniciar_agente(tamanho=10)

        while ambiente.simulacao_ativa:
            for agente in agentes:
                ambiente.start_time = time.time()  
                try:
                    acao_escolhida = agente.explorar_inteligentemente(ambiente)
                except TypeError:
                    pass
                ambiente.mover_agente(agente, acao_escolhida, ambiente)
                time.sleep(0.2)
                ambiente.end_time = time.time()  
                # Atualize os tempos_execucao
                ambiente.tempo_execucao_agente = (ambiente.end_time - ambiente.start_time)/60
                ambiente.tempos_execucao.append(ambiente.tempo_execucao_agente)

def run_RForest():
    if __name__ == "__main__":
            threading.Thread(target=Modelo_RForest.run).start()

class Render: 
    def renderizar_Abordagem_A():
        ambiente = Ambiente(tamanho=10, probabilidade_livre=0, probabilidade_bomba=0, probabilidade_tesouro=0)
        print('================== EXCUTANDO ABORDAGEM--A ====================')
        print('==============================================================')
        print('Considerar sucesso se o total de Tesouros descobertos for acima de 50% dos Tesouros semeados no ambiente')
        ambiente.desenhar_ambiente()