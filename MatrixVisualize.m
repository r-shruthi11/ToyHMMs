A = randi(3,3);
A_ = randi(3,3);

%% Heatmap
close all;
f = figure();
subplot(2,1,1);
imagesc(A)
subplot(2,1,2);
imagesc(A_)

%% Plot
close all;
g = figure();
plot(A, 'b');
hold on;
plot(A_, 'r');

%% RMS
close all;
dA = A(:)-A_(:);
dArms = sqrt(mean(power(dA, 2)));
plot(dArms);

%% L1 norm
close all;
dA1 = sum(abs(A(:)-A_(:)));
plot(dA1);

