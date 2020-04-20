X = data;

X1 = X(:, 1:end -1) ;
X2 = X(:, 2: end );
%% SVD and rank -2 truncation
r = 107; % rank truncation
[U, S, V] = svd (X1);
Ur = U(:, 1:r);
Sr = S(1:r, 1:r);
Vr = V(:, 1:r);
%% Build Atilde and DMD Modes
Atilde = Ur' *X2*Vr/Sr;
[W, D] = eig (Atilde );
Phi = X2*Vr/Sr*W; % DMD Modes
%% DMD Spectra
lambda = diag (D);
omega = log ( lambda );
%% Compute DMD Solution
x1 = X(:, 1);
b = Phi \x1;
mm1 = size(X1, 2);
t = (0:mm1-1);
time_dynamics = zeros(r, length (t));
for iter = 1: length (t),
time_dynamics (:, iter ) = (b.*exp ( omega*t(iter )));
end ;
X_dmd = Phi * time_dynamics;
plot(X2(42,:));

plot(real(X_dmd(42,:)));

