% SEL0362 - Inteligência Artificial
% Exercício 3
% Felipe Pimenta Bernardo - 10788697
% Henrique Sander Lourenço - 10802705


clear all 
close all 
clc

% Entrada com os 4 primeiros quadrados marcados incorretamente
X = [1 1 1 1 0 0 0 0 1 1 1 1 1 1 1 1];
            
load('W1');
load('W2');

y = zeros(1, 3); 
v1 = W1 * X';
y1 = sigmoid(v1);
v = W2 * y1;
y = sigmoid(v); 
disp("Saída para entrada com 4 quadrados marcados incorretamente:");
disp(y);