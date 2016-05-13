function [part1, part2] = divideByRow(set, ratio1, ratio2, num_labels) 
    
    label_block_size = size(set, 1) / num_labels;
    label_fract_size = label_block_size / (ratio1 + ratio2);
    part1 = [];
    part2 = [];

    for i = 1:num_labels,
        st_pos = ( label_block_size * (i - 1) );
        part1 = [part1 ; set(st_pos + 1: st_pos + label_fract_size * ratio1, :)];
        part2 = [part2 ; set(st_pos + label_fract_size * ratio1 + 1 : st_pos + label_block_size, :)];
    end
end
