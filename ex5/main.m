% SEL0362 - Inteligencia Artificial
% Exercicio 5
% Felipe Pimenta Bernardo - 10788697
% Henrique Sander Lourenco - 10802705

% Entradas: 
% 1 = vaso; 0 = fundo

% Entrada 1: vaso preto
X = [1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0;
     1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0;
     1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0;
     0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];

% Entrada 2: vaso marrom cafe
X(:, :, 2) = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
              0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0;
              0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0;
              0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0;
              0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0];

% Entrada 3: vaso areia
X(:, :, 3) = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
              0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0;
              0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0;
              0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0;
              0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0];

% Entrada 4: vaso marrom chocolate
X(:, :, 4) = [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1;
              0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1;
              0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1;
              0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0;
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];

% Saida desejada
D = [1 0 0 0;   % Vaso preto
     0 1 0 0;   % Vaso marrom cafe
     0 0 1 0;   % Vaso areia
     0 0 0 1];  % Vaso marrom chocolate

nusp = [1 0 8 0 2 7 0 5]; % NUSP do aluno

% nmro neuronios 1a camada escondida
hiddenLayer1_neurons = sum(nusp); 
% nmro neuronios 2a camada escondida
hiddenLayer2_neurons = ceil(hiddenLayer1_neurons / 2);

inputSize = size(X, 1) * size(X, 2); % tamanho entrada
outputSize = size(D, 1); % tamanho saida

% Matrizes de pesos
W1 = 2 * rand(hiddenLayer1_neurons, inputSize) - 1;
W2 = 2 * rand(hiddenLayer2_neurons, hiddenLayer1_neurons) - 1;
W3 = 2 * rand(outputSize, hiddenLayer2_neurons) - 1;

% Treinamento
epoch = 1000;
mse = zeros(1, epoch);
for i = 1:epoch
    [W1, W2, W3, mse(i)] = DeepReLU(W1, W2, W3, X, D);
end

plot(mse);
grid;
title('Erro quadratico medio por treinamento');
xlabel('Treinamento');
ylabel('Erro quadratico medio');

% Inferencia
N = size(X, 3);
Y = zeros(size(D, 1), size(D, 2)); 
for k = 1:N
    x = reshape(X(:, :, k), size(X, 1) * size(X, 2), 1);
    v1 = W1 * x;
    y1 = max(0, v1); % Funcao unitaria linear retificada (ReLU)
    v2 = W2 * y1;
    y2 = max(0, v2); % Funcao unitaria linear retificada (ReLU)
    v3 = W3 * y2;
    y3 = softmax(v3);
    Y(k, :) = y3;
end

disp('    Saida desejada                          Saida obtida');
disp([D Y]);

% Entradas corrompidas
X_teste = [1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0;
           1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0;
           0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0;
           0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
           0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
X_teste(:, :, 2) = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
                    0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0;
                    0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0;
                    0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0;
                    0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0];
X_teste(:, :, 3) = [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
                    0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0;
                    0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0;
                    0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0;
                    0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0];
X_teste(:, :, 4) = [0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1;
                    0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1;
                    0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1;
                    0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1;
                    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];

% Inferencia
Y_teste = zeros(size(D, 1), size(D, 2)); 
for k = 1:N
    x = reshape(X_teste(:, :, k), size(X, 1) * size(X, 2), 1);
    v1 = W1 * x;
    y1 = max(0, v1); % Funcao unitaria linear retificada (ReLU)
    v2 = W2 * y1;
    y2 = max(0, v2); % Funcao unitaria linear retificada (ReLU)
    v3 = W3 * y2;
    y3 = softmax(v3);
    Y_teste(k, :) = y3;
end

disp('    Saida desejada                          Saida obtida');
disp([D Y_teste]);