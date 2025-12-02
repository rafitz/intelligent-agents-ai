!pip install scikit-fuzzy

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

qualidade = ctrl.Antecedent(np.arange(0, 11, 1), 'qualidade')
servico = ctrl.Antecedent(np.arange(0, 11, 1), 'servico')

gorjeta = ctrl.Consequent(np.arange(0, 21, 1), 'gorjeta')

qualidade.automf(number=3, names=['ruim', 'boa', 'saborosa'])
servico.automf(number=3, names=['ruim', 'aceitável', 'ótimo'])

gorjeta['baixa'] = fuzz.sigmf(gorjeta.universe, 5, -1)
gorjeta['media'] = fuzz.gaussmf(gorjeta.universe, 10, 3)
gorjeta['alta'] = fuzz.pimf(gorjeta.universe, 10, 20, 20, 21)

# Definição das 20 regras
regras = [
    ctrl.Rule(qualidade['ruim'] | servico['ruim'], gorjeta['baixa']),
    ctrl.Rule(servico['aceitável'], gorjeta['media']),
    ctrl.Rule(servico['ótimo'] & qualidade['saborosa'], gorjeta['alta']),
    ctrl.Rule(qualidade['ruim'] & servico['aceitável'], gorjeta['baixa']),
    ctrl.Rule(qualidade['ruim'] & servico['ótimo'], gorjeta['media']),
    ctrl.Rule(qualidade['boa'] & servico['ruim'], gorjeta['baixa']),
    ctrl.Rule(qualidade['boa'] & servico['aceitável'], gorjeta['media']),
    ctrl.Rule(qualidade['boa'] & servico['ótimo'], gorjeta['alta']),
    ctrl.Rule(qualidade['saborosa'] & servico['ruim'], gorjeta['media']),
    ctrl.Rule(qualidade['saborosa'] & servico['aceitável'], gorjeta['alta']),
    ctrl.Rule(servico['ruim'], gorjeta['baixa']),
    ctrl.Rule(qualidade['ruim'], gorjeta['baixa']),
    ctrl.Rule(servico['ótimo'], gorjeta['alta']),
    ctrl.Rule(qualidade['saborosa'], gorjeta['alta']),
    ctrl.Rule(qualidade['boa'], gorjeta['media']),
    ctrl.Rule(servico['ótimo'] | qualidade['saborosa'], gorjeta['alta']),
    ctrl.Rule(servico['aceitável'] & qualidade['boa'], gorjeta['media']),
    ctrl.Rule(servico['aceitável'] & qualidade['ruim'], gorjeta['baixa']),
    ctrl.Rule(servico['ótimo'] & qualidade['boa'], gorjeta['alta']),
    ctrl.Rule(servico['ruim'] & qualidade['saborosa'], gorjeta['media'])
]

sistema_controle = ctrl.ControlSystem(regras)
sistema = ctrl.ControlSystemSimulation(sistema_controle)

# Você pode alterar estes valores para testar outros cenários
sistema.input['qualidade'] = 10
sistema.input['servico'] = 10
sistema.compute()

# Impressão do resultado e visualização
print(f"Valor da Gorjeta Calculada: {sistema.output['gorjeta']:.2f}%")
gorjeta.view(sim=sistema)
plt.show()