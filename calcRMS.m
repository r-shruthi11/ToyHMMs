function [rmsMAT] = calcRMS(A, B)

rmsMAT = sqrt(mean(power(A(:)-B(:), 2)));

end