% Test optimizableVariable syntax
clc;

try
    % Test 1: Basic optimizableVariable
    var1 = optimizableVariable('learning_rate', [1e-5, 1e-2], 'Transform', 'log');
    fprintf('Test 1 passed: var1 created\n');
    fprintf('  Name: %s\n', var1.Name);
    fprintf('  Range: [%g, %g]\n', var1.Range(1), var1.Range(2));
    fprintf('  Type: %s\n', var1.Type);
    fprintf('  Transform: %s\n', var1.Transform);

    % Check if Optimize field exists
    if isprop(var1, 'Optimize')
        fprintf('  Optimize: %d\n', var1.Optimize);
    else
        fprintf('  Optimize property does not exist\n');
    end

    % Test 2: Integer type
    var2 = optimizableVariable('embedding_dim', [32, 256], 'Type', 'integer');
    fprintf('\nTest 2 passed: var2 created\n');

    % Create array
    optimVars = [var1, var2];
    fprintf('Array created with %d variables\n', numel(optimVars));

    % Test dot indexing
    fprintf('\nTesting dot indexing on array:\n');
    for i = 1:numel(optimVars)
        fprintf('  Var %d: Name=%s, Range=[%g, %g]\n', ...
            i, optimVars(i).Name, optimVars(i).Range(1), optimVars(i).Range(2));
    end

    % Test Optimize indexing (if property exists)
    if isprop(optimVars(1), 'Optimize')
        fprintf('\nOptimize values: ');
        fprintf('%d ', [optimVars.Optimize]);
        fprintf('\n');
    end

catch ME
    fprintf('Error: %s\n', ME.message);
end