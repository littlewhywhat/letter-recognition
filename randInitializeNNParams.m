function nn_params = randInitializeNNParams(nn_structure)
    nn_params = [];
    for i = 2:length(nn_structure),
        initial_Thetai = randInitializeWeights(nn_structure(i-1), nn_structure(i));
        % repeat check
        [mat, index] = unique(initial_Thetai, 'rows', 'first');
        repeatedIndex = setdiff(1:size(initial_Thetai, 1),index);
        if (size(repeatedIndex, 2) != 0)
            fprintf('\n Collision!\n');
        endif
        nn_params = [nn_params ; initial_Thetai(:)];
    end
end
