function [part1, part2] = divideData(set, ratio1, ratio2) 
    sel = randperm(size(set,1));
    part1_size = size(set,1)/(ratio1 + ratio2) * ratio1; 
    part1 = set(sel(1 : part1_size), :);
    part2 = set(sel(part1_size + 1 : end), :);
end
