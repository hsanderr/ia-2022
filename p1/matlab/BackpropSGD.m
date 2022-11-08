% SEL0362 - Inteligência Artificial
% Prova 1
% Henrique Sander Lourenço - 10802705

% Treinamento da rede neural com gradiente descendente estocástico por backpropagation

function [W1, W2, meq] = BackpropSGD(W1, W2, X, D, numeroEntradas)
    alpha = 0.9; % Taxa de aprendizagem
    acc = 0; % Acumulador para cálculo do erro quadrático mínimo
    
    for k = 1:numeroEntradas
        x = X(k, :)'; % Entrada k da rede neural
        d = D(k, :)'; % Saída desejada para respectiva entrada
        v1 = W1 * x; 
        y1 = sigmoid(v1); % Saída dos nós da camada escondida
        v  = W2 * y1;
        y = sigmoid(v); % Saída dos nós da camada de saída
        e = d - y; % Erro na saída da rede neural
        acc = acc + min(e.^2); % Cálculo do erro quadrático mínimo
        delta = y .* (1 - y) .* e;
        e1 = W2' * delta; % Erro na saída dos nós da camada escondida
        delta1 = y1 .* (1 - y1) .* e1;

        % Atualização dos pesos
        dW1 = alpha * delta1 * x';        
        W1 = W1 + dW1;        
        dW2 = alpha * delta * y1';
        W2 = W2 + dW2;

    end
    meq = acc / numeroEntradas; % Erro quadrático mínimo
end