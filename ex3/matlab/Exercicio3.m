% SEL0362 - Inteligência Artificial
% Exercício 3
% Felipe Pimenta Bernardo - 10788697
% Henrique Sander Lourenço - 10802705


clear all 
close all 
clc

% The input set consists of three 4x4 block image;
% User must select the blocks that match the description;
% 0: do not match; 1: do match. 

% Entrada: 3 vetores de 16 posições indicando os blocos do captcha
% 1: correspondência
% 0: não correspondência
X = [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1; % Placas de trânsito
     1 1 1 1 0 0 1 1 0 0 0 0 0 0 0 0; % Veículos
     0 0 1 0 0 1 1 1 0 0 0 0 0 0 0 0]; % Semáforos
            
% Saída desejada: uma saída para cada captcha
% 1: resposta correta
% 0: resposta incorreta
D = [1 0 0;
     0 1 0;    
     0 0 1];

somaNusp = 23; % Soma NUSP (10802705)
            
% Inicialização dos pesos da camada escondida
% número de nós da camada escondida x número de entradas
W1 = 2 * rand(somaNusp, 16) - 1; 
% Inicialização dos pesos da camada de saída
% número de saídas x número de nós da camada escondida
W2 = 2 * rand(3, somaNusp) - 1; % Pesos da camada saída: 3

% Treinamento por back-propagation SGD:
epoca = 100;
meq = zeros(1, epoca); % Inicialização do erro quadrático mínimo
for i = 1:epoca
    [W1, W2, meq(i)] = BackpropSGD(W1, W2, X, D);
end

% Inferência
numeroEntradas = 3;
y = zeros(numeroEntradas, size(D,2)); 
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

figure(1)
plot(1:1:epoca, meq, 'LineWidth', 1.5);
title('Convergência do erro de treinamento');
xlabel('Época');
ylabel('Erro quadrático mínimo');
xlim([1 i]); grid on;