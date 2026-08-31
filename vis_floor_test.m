cd /Users/ellag/Desktop/PhD/academic_projects/tang_bss_schooling/data/floor_test

%Load in data
coord_filename = "/Users/ellag/Desktop/PhD/academic_projects/tang_bss_schooling/data/floor_test/redband_123024_1 _249_sl4_individual (1).csv" %ADD NAME OF BALL DROP FILE WITH 3D COORDS
imgcoordsRC = readtable(coord_filename);

figure
hold on
grid on

for k = 1:size(imgcoordsRC,1)
    % Extract coordinates for base, head, and ball points
    base_x = imgcoordsRC{k,4};
    base_y = -imgcoordsRC{k,5}; %invert y for vis because gui annotation 0,0 is top left
    base_z = imgcoordsRC{k,8};
    
    head_x = imgcoordsRC{k,6};
    head_y = -imgcoordsRC{k,7};
    head_z = imgcoordsRC{k,9};

    %ball_first_x = imgcoordsRC{1,50};
    %ball_first_y = imgcoordsRC{1,51};
    %ball_first_z = imgcoordsRC{1,52};
    
    %ball_drop_x = imgcoordsRC{1,80};  % ball_hit_X
    %ball_drop_y = imgcoordsRC{1,81};  % ball_hit_Y
    %ball_drop_z = imgcoordsRC{1,82};  % ball_hit_Z
    % 
    % Skip if any key coordinates are missing
    % if any(isnan([base_x, base_y, base_z, head_x, head_y, head_z, ...
    %               ball_first_x, ball_first_y, ball_first_z, ...
    %               ball_drop_x, ball_drop_y, ball_drop_z]))
    %     continue
    % end

    %    Line from base to head
    plot3([base_x, head_x], [base_y, head_y], [base_z, head_z], '-b')

    % %Plot head point (green)
    scatter3(head_x, head_y, head_z, 15, [0, 0.6, 0], 'filled')

    % --- Plot the eel base ---
    scatter3(base_x, base_y, base_z, 15, [0.65, 0.16, 0.16], 'filled') % brown base

    % --- Plot the ball drop as a pink star ---
    %scatter3(ball_drop_x, ball_drop_y, ball_drop_z, 100, 'p', 'MarkerEdgeColor', 'k', ...
    %         'MarkerFaceColor', [1, 0.4, 0.7]) % pink star
    
    % %--- Line from ball drop to first view ---
    % plot3([ball_drop_x, ball_first_x], [ball_drop_y, ball_first_y], [ball_drop_z, ball_first_z], ...
    %     '-m', 'LineWidth', 1.5) % magenta line

    % --- Add label (optional) ---
    text(base_x, base_y, base_z, imgcoordsRC{k,3}, ...
         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', ...
         'Color', 'b', 'FontSize', 6)

end

xlabel('X'); ylabel('Y'); zlabel('Z');
title('PC Ball Drop and Eel 3D Visualization');
axis equal
hold off
