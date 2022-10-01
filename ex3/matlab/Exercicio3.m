% SEL0362 - Intelig�ncia Artificial
% Exerc�cio 3
% Felipe Pimenta Bernardo - 10788697
% Henrique Sander Louren�o - 10802705


clear all 
close all 
clc

% The input set consists of three 4x4 block image;
% User must select the blocks that match the description;
% 0: do not match; 1: do match. 

% Entrada: 3 vetores de 16 posi��es indicando os blocos do captcha
% 1: correspond�ncia
% 0: n�o correspond�ncia
X = [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1; % Placas de tr�nsito
     1 1 1 1 0 0 1 1 0 0 0 0 0 0 0 0; % Ve�culos
     0 0 1 0 0 1 1 1 0 0 0 0 0 0 0 0]; % Sem�foros
            
% Sa�da desejada: uma sa�da para cada captcha
% 1: resposta correta
% 0: resposta incorreta
D = [1 0 0;
     0 1 0;    
     0 0 1];

somaNusp = 23; % Soma NUSP (10802705)
            
% Inicializa��o dos pesos da camada escondida
% n�mero de n�s da camada escondida x n�mero de entradas
W1 = 2 * rand(somaNusp, 16) - 1; 
% Inicializa��o dos pesos da camada de sa�da
% n�mero de sa�das x n�mero de n�s da camada escondida
W2 = 2 * rand(3, somaNusp) - 1; % Pesos da camada sa�da: 3

% Treinamento por back-propagation SGD:
epoca = 100;
meq = zeros(1, epoca); % Inicializa��o do erro quadr�tico m�nimo
for i = 1:epoca
    [W1, W2, meq(i)] = BackpropSGD(W1, W2, X, D);
end

% Infer�ncia
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
disp('               [sa�da desejada]  [sa�da obtida]'); 
disp([D y])

figure(1)
plot(1:1:epoca, meq, 'LineWidth', 1.5);
title('Converg�ncia do erro de treinamento');
xlabel('�poca');
ylabel('Erro quadr�tico m�nimo');
xlim([1 i]); grid on;