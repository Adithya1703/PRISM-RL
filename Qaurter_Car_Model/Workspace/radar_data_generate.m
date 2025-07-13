v = 2;
d_preview = 1.0; % radar preview distance[m]

% generate radar data
zr_future = simulateRadarPreview(road_profile_time, road_profile_height, v, d_preview);

% plot comparision
figure;
plot(road_profile_time, road_profile_height, 'b', 'DisplayName', 'True road');
hold on
plot(road_profile_time, zr_future, 'r--', 'DisplayName', 'Radar Preview');
xlabel('Time [s]');
ylabel('ROad Elevation [m]');
legend;
title('Simulated Radar Road Preview');
grid on;
saveas(gcf, 'radar_vs_road.png');

t_col = road_profile_time(:);
zr_col = zr_future(:);

% combine into a 2 column matrix
% Open file
fid = fopen('radar_data.csv', 'w');

% Write headers
fprintf(fid, 'Time (s),Radar road elevation (m)\n');

% Write data row by row
for i = 1:length(road_profile_time)
    fprintf(fid, '%.6f,%.6f\n', road_profile_time(i), zr_future(i));
end

fclose(fid);