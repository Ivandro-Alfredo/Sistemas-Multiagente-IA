  
from collections import Counter

def votar(previsoes):
    votos = Counter(previsoes)
    previsao_mais_comum = votos.most_common(1)[0][0]
    return previsao_mais_comum
    if previsao_mais_comum >= 0:
        return previsao_mais_comum
    else:
        return None
    
previsoes = [1, 2, 0]

if all(previsoes[i] != previsoes[j] for i in range(len(previsoes)) for j in range(i + 1, len(previsoes))):
    print('Voltou: ', votar(previsoes))

# def todos_diferentes(lista):
#     return all(lista[i] != lista[j] for i in range(len(lista)) for j in range(i + 1, len(lista)))

# previsoes = [11, 1, 3]
# print('Todos diferentes? ', todos_diferentes(previsoes))  # Retorna True, pois todos os elementos são diferentes

# previsoes = [1, 1, 2, 3]
# print('Todos diferentes?', todos_diferentes(previsoes))  # Retorna False, pois há elementos iguais na lista


    