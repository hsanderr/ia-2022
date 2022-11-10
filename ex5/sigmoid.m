%%  Definição da função sigmoide
function sigmoid = sigmoid(x)
     sigmoid = 1 ./ (1 + exp(-x));
end 