% Run CheckEM_MultiRuns_InitialRand for N = 1000 many times and visualize
% the results for each

nTimes = 10;

for nt = 1:nTimes
    
    [params, Afinal, Phifinal] = CheckEMSingleRun_InitialRand();
    
    f = figure('units','normalized','outerposition',[0 0 1 1]);
    
    subplot(2,1,1);
    
%     imagesc(reshape(Phiestimates(5).Phiestim(:, end), [3,4]));

    imagesc(Phifinal);
    
    title('Final estimate of Phi');

    subplot(2,1,2);

    imagesc(params.Phi);
    
    title('Model parameter Phi');
    
    saveas(f, ['./Figs/PhiComp' num2str(nt)], 'epsc');
    
    
    g = figure('units','normalized','outerposition',[0 0 1 1]);
    
    subplot(2,1,1);
    
%     imagesc(reshape(Phiestimates(5).Phiestim(:, end), [3,4]));

    imagesc(Afinal);
    
    title('Final estimate of A');

    subplot(2,1,2);

    imagesc(params.A);
    
    title('Model parameter A');
    
    saveas(g, ['./Figs/AComp' num2str(nt)], 'epsc');
    
end
    
    
    
    

