import numpy as np
from rbm import RBM


base = np.array([[1,1,1,0,0,0], [1,0,1,0,0,0], [1,1,1,0,0,0],
                 [0,0,1,1,1,1], [0,0,1,1,0,1], [0,0,1,1,0,1]])


    ### Criação do modelo de Recomendação ###
rbm = RBM(num_visible=6, num_hidden=2)

    ### Treino do modelo de Recomendação ###
rbm.train(base, max_epochs=5000)
rbm.weights
 # valores na 1º linha e na 1º coluna são unidades de bias
 # demais valores indicam os pesos
 # valores positivos indicam ativação do neurônio





        # =====   Recomendação de Novo Registro   ===== #
novo_registro = np.array([[1,1,0,1,0,0], [0,0,0,1,1,0]])
activated_neurons = rbm.run_visible(novo_registro) # indica qual neuronio foi ativado
recomendacao = rbm.run_hidden(activated_neurons)


filmes = ['A Bruxa', 'invocação do Mal', 'O chamado', 
          'Se Beber não Case', 'Gente Grande', 'American Pie']

for u in range(novo_registro.shape[0]):
    print('\nRecomendação para Usuário {}'.format(u+1))
    for f in range(novo_registro.shape[1]):
        if novo_registro[u, f] == 0 and recomendacao[u, f] == 1:
            print(' - {}'.format(filmes[f]))