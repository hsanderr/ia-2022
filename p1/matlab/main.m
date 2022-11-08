% SEL0362 - Inteligência Artificial
% Prova 1
% Henrique Sander Lourenço - 10802705

clear all 
close all 
clc

% Entradas: 3 vetores de 16 posicoes indicando os criterios de viabilidade
% 1: Viavel (V)
% 0: Nao viavel (NV)
X = [1 1 0 0 0 0 1 0 0 0 1 0 0 0 1 0; % SC
     1 0 1 0 0 0 1 0 1 0 1 0 0 0 0 0; % RJ
     1 1 1 1 0 0 1 0 0 0 1 0 1 1 1 1]; % SP
            
% Saidas:
% 1: indica o estado correspondente
% 1a posicao: SC
% 2a posicao: RJ
% 3a posicao: SP
D = [1 0 0;
     0 1 0;    
     0 0 1];

somaNusp = 23; % Soma NUSP (10802705)
numeroEntradas = 3; % Numero de entradas
tamanhoEntrada = 16; % Tamanho do vetor de entrada
numeroSaidas = 3; % Numero de saidas
tamanhoSaida = 3; % Tamanho do vetor de saida
            
% Inicializacao dos pesos da camada escondida
% numero de nos da camada escondida x tamanho das entradas
W1 = 2 * rand(somaNusp, tamanhoEntrada) - 1; 

% Inicializacao dos pesos da camada de saida
% tamanho das saidas x numero de nos da camada escondida
W2 = 2 * rand(tamanhoSaida, somaNusp) - 1;

% Treinamento por back-propagation SGD:
epoca = 500;
meq = zeros(1, epoca); % Inicializacao do erro quadratico minimo
for i = 1:epoca
    [W1, W2, meq(i)] = BackpropSGD(W1, W2, X, D, numeroEntradas);
end

W1
W2

% Inferencia
y = zeros(numeroSaidas, tamanhoSaida); 
for k = 1:numeroEntradas
    x = X(k, :)';
    v1 = W1 * x;
    y1 = sigmoid(v1);
    v = W2 * y1;
    y(k, :) = sigmoid(v);   
end

disp('Resultados:');
disp('               [saída desejada]  [saída obtida]'); 
disp([D y])

% Para cada saida, indica se foi possivel identificar
% o estado e, se sim, qual estado foi identificado
for j = 1:3
    z = y(j, :);
    % Flag para saber se o estado foi identificado
    % 1: nao identificado
    % 0: identificado
    estadoNaoIdentificado = 0;
    for i = 1:3
        % Se algum elemento da saida ficar entre 0.1 e 0.9,
        % nao foi possivel identificar o estado
        if z(i) < 0.9 && z(i) > 0.1
            estadoNaoIdentificado = 1;
            disp("Não foi possível identificar o estado");
            break
        end
    end
    
    % Se o estado foi identificado, verifica qual deles foi
    if estadoNaoIdentificado == 0
        if z(1) >= 0.9 && z(2) <= 0.1 && z(3) <= 0.1
            disp("Estado: SC")
        elseif z(2) >= 0.9 && z(1) <= 0.1 && z(3) <= 0.1
            disp("Estado: RJ")
        elseif z(3) >= 0.9 && z(1) <= 0.1 && z(2) <= 0.1
            disp("Estado: SP")
        % Caso nenhum ou mais de um elemento da saida apresente valor
        % maior que 0.9, a identificacao falhou
        else
            disp("Erro desconhecido ao identificar a imagem")
        end
    end
end

figure(1)
plot(1:1:epoca, meq, 'LineWidth', 1.5);
title('Convergência do erro de treinamento');
xlabel('Época');
ylabel('Erro quadrático mínimo');
xlim([1 epoca]);
grid on;