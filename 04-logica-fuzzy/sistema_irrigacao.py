!pip install scikit-fuzzy

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 2. Definição das Variáveis (Antecedentes e Consequente)

# Universo de discurso para Umidade do Solo (0% a 100%)
universo_umidade = np.arange(0, 101, 1)
# Universo de discurso para Temperatura (0°C a 40°C)
universo_temp = np.arange(0, 41, 1)
# Universo de discurso para Tempo de Rega (0 a 60 minutos)
universo_tempo_rega = np.arange(0, 61, 1)

# Criando as variáveis fuzzy
umidade = ctrl.Antecedent(universo_umidade, 'umidade')
temperatura = ctrl.Antecedent(universo_temp, 'temperatura')
tempo_rega = ctrl.Consequent(universo_tempo_rega, 'tempo_rega')

# 3. Definição das Funções de Pertinência (Membership Functions)

# Gerando MFs automáticas para as entradas
umidade.automf(number=3, names=['seco', 'úmido', 'encharcado'])
temperatura.automf(number=3, names=['fria', 'morna', 'quente'])

# Gerando MFs manuais (triangulares) para a saída
tempo_rega['curto'] = fuzz.trimf(tempo_rega.universe, [0, 0, 20])
tempo_rega['médio'] = fuzz.trimf(tempo_rega.universe, [15, 30, 45])
tempo_rega['longo'] = fuzz.trimf(tempo_rega.universe, [40, 60, 60])



# 4. Definição das Regras Fuzzy
# ( | é 'OU', & é 'E' )
regra1 = ctrl.Rule(umidade['encharcado'] | temperatura['fria'], tempo_rega['curto'])
regra2 = ctrl.Rule(umidade['úmido'], tempo_rega['médio'])
regra3 = ctrl.Rule(umidade['seco'] & temperatura['quente'], tempo_rega['longo'])

# 5. Criação do Sistema de Controle
sistema_controle_irrigacao = ctrl.ControlSystem([regra1, regra2, regra3])
sistema_irrigacao = ctrl.ControlSystemSimulation(sistema_controle_irrigacao)

# 6. Simulação (Defuzzificação)

# Vamos simular um dia muito seco (ex: 15%) e quente (ex: 35°C)
sistema_irrigacao.input['umidade'] = 15
sistema_irrigacao.input['temperatura'] = 35

# Computando o resultado
sistema_irrigacao.compute()

# 7. Exibição do Resultado
print("\n--- Resultado da Simulação ---")
print(f"Entrada Umidade: 15%")
print(f"Entrada Temperatura: 35°C")
print(f"Tempo de Rega Calculado: {sistema_irrigacao.output['tempo_rega']:.2f} minutos")

# Visualizando o resultado da simulação
print("Gráfico do Resultado da Defuzzificação:")
tempo_rega.view(sim=sistema_irrigacao)

# Visualizando as Funções de Pertinência
print("Funções de Pertinência da Umidade:")
umidade.view()

print("Funções de Pertinência da Temperatura:")
temperatura.view()

print("Funções de Pertinência do Tempo de Rega:")
tempo_rega.view()