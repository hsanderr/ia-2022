% SEL0362 - Inteligencia Artificial
% Exercicio 3
% Felipe Pimenta Bernardo - 10788697
% Henrique Sander Lourenço - 10802705

clear all 
close all 
clc

X = zeros(1, 16); % Inicializa entrada

% Pega entrada do usuario
for i = 1:16
    prompt = strcat("Insira o elemento ", num2str(i), ...
        " do vetor de entrada: ");
    X(i) = input(prompt);
    if X(i) ~= 0 && X(i) ~= 1
        disp("A entrada deve ser 0 (zero) ou 1 (um)")
        return
    end
end

load('W1');
load('W2');

% Inferencia
y = zeros(1, 3); 
v1 = W1 * X';
y1 = sigmoid(v1);
v = W2 * y1;
y = sigmoid(v); 
disp("Saida: ");
disp(y);

dummy = 0; % Flag para identificar se usuario e um dummy
for i = 1:3
    if y(i) < 0.9 && y(i) > 0.1
        dummy = 1;
        disp("Usuario e um robo");
        break
    end
end

if dummy == 0
    disp("Usuario e um humano");
    if y(1) >= 0.9 && y(2) <= 0.1 && y(3) <= 0.1
        disp("Imagem: placas de transito")
    elseif y(2) >= 0.9 && y(1) <= 0.1 && y(3) <= 0.1
        disp("Imagem: veiculos")
    elseif y(3) >= 0.9 && y(1) <= 0.1 && y(2) <= 0.1
        disp("Imagem: semaforos")
    else
        disp("Erro desconhecido ao identificar a imagem")
    end
end
    

