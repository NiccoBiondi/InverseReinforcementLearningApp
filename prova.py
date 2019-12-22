lista = ['clipsToAnnotate_0', 'clipsToAnnotate_1', 'clipsToAnnotate_2']


index = [i for i in range(len(lista)) if '2' in lista[i]][0]
listas = [lista[i] for i in range([i for i in range(len(lista)) if '2' in lista[i]][0], len(lista))] 

print(listas)