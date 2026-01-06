% Simple test of bayesopt
clc; clear;

% Simple objective function
fun = @(x) (x.x1 - 0.3)^2 + (x.x2 - 0.7)^2;

% Create optimizable variables
x1 = optimizableVariable('x1', [0, 1]);
x2 = optimizableVariable('x2', [0, 1]);
vars = [x1, x2];

fprintf('Testing bayesopt with simple function...\n');
try
    results = bayesopt(fun, vars, ...
        'MaxObjectiveEvaluations', 5, ...
        'Verbose', 1);
    fprintf('Success! Best point:\n');
    disp(results.XAtMinObjective);
catch ME
    fprintf('Error: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s line %d\n', ME.stack(i).name, ME.stack(i).line);
    end
end