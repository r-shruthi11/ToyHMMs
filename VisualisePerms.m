close all;

figure('Name', 'Phi');

subplot(3,1,1);

imagesc(reshape(Phiestimates(5).Phiestim(:, end), [3,4]));

subplot(3,1,2);

imagesc(params.Phi);

subplot(3,1,3);

colorbar();

figure('Name', 'A');

subplot(2,1,1);

imagesc(reshape(Aestimates(5).Aestim(:, end), [3,3]));

subplot(2,1,2);

imagesc(params.A);

colorbar();
