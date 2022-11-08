% SEL0362 - Intelig�ncia Artificial
% Prova 1
% Henrique Sander Louren�o - 10802705

% Treinamento da rede neural com gradiente descendente estoc�stico por backpropagation

function [W1, W2, meq] = BackpropSGD(W1, W2, X, D, numeroEntradas)
    alpha = 0.9; % Taxa de aprendizagem
    acc = 0; % Acumulador para c�lculo do erro quadr�tico m�nimo
    
    for k = 1:numeroEntradas
        x = X(k, :)'; % Entrada k da rede neural
        d = D(k, :)'; % Sa�da desejada para respectiva entrada
        v1 = W1 * x; 
        y1 = sigmoid(v1); % Sa�da dos n�s da camada escondida
        v  = W2 * y1;
        y = sigmoid(v); % Sa�da dos n�s da camada de sa�da
        e = d - y; % Erro na sa�da da rede neural
        acc = acc + min(e.^2); % C�lculo do erro quadr�tico m�nimo
        delta = y .* (1 - y) .* e;
        e1 = W2' * delta; % Erro na sa�da dos n�s da camada escondida
        delta1 = y1 .* (1 - y1) .* e1;

        % Atualiza��o dos pesos
        dW1 = alpha * delta1 * x';        
        W1 = W1 + dW1;        
        dW2 = alpha * delta * y1';
        W2 = W2 + dW2;

    end
    meq = acc / numeroEntradas; % Erro quadr�tico m�nimo
end