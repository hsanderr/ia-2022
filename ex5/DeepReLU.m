% SEL0362 - Inteligencia Artificial
% Exercicio 5
% Felipe Pimenta Bernardo - 10788697
% Henrique Sander Lourenco - 10802705

% Aprendizado profundo com funcao unitaria linear retificada (ReLU),
% backpropagation e entropia cruzada
function [W1, W2, W3, errAvg] = DeepReLU(W1, W2, W3, X, D) 
    alpha = 0.01; % Taxa de aprendizagem
    N = size(X, 3); % Numero de amostras
    sampleErrAvg = zeros(1, N);
    for k = 1:N
        x = reshape(X(:, :, k), size(X, 1) * size(X, 2), 1);
        v1 = W1 * x;
        y1 = max(0, v1); % Funcao unitaria linear retificada (ReLU)
        v2 = W2 * y1;
        y2 = max(0, v2); % Funcao unitaria linear retificada (ReLU)
        v3 = W3 * y2;
        y3 = softmax(v3);
        d = D(k, :)';
        sampleErr = d - y3;
        delta3 = sampleErr;
        e2 = W3' * delta3;
        delta2 = (v2 > 0) .* e2; 
        e1 = W2' * delta2;
        delta1 = (v1 > 0) .* e1;

        % Erro quadrático médio da amostra
        sampleErrAvg(k) = sum(sampleErr .^ 2) / size(sampleErr, 1);
        
        % Ajuste dos pesos
        dW3 = alpha * delta3 * y2'; 
        W3 = W3 + dW3;
        dW2 = alpha * delta2 * y1'; 
        W2 = W2 + dW2;
        dW1 = alpha * delta1 * x';
        W1 = W1 + dW1;
    end

    % Erro quadrático médio dos erros quadráticos médios das amostras
    errAvg = sum(sampleErrAvg .^ 2) / N;
end
    