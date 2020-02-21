function [C, sigma] = optimizeSVM(X, y, kernelFunction)
% Custom-built function attempting to describe the optimization process for finding sigma and C.

% TODO: try: function [C, sigma] = optimizeSVM(X, y, kernelFunction, factors)
	factors = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
	
	for i = 1:length(factors)
		C = factors(i);
		for j = 1:length(factors)
			sigma = factors(j);
			
		end
	end


end