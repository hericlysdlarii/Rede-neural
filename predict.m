function p = predict(Theta1, Theta2, X)
%  A função predict é responsável por predizer uma label da imagem de entrada X,
%  dada as matrizes de pesos Theta1 e Theta2. 

% Inicializando os valores das variáveis

m = size(X, 1);
%num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);

% não esqueça de add os bias das camadas de ativação
a1 = [ones(m,1), X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
aux = [ones(m,1)];
aux(:,:) = 2;
a2 = [aux, a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

[val, p] = max(a3, [], 2);


end
