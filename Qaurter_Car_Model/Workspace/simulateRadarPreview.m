function zr_future = simulateRadarPreview(road_time, road_height, vehicle_speed, preview_distance)
% Simulate mmWave radar preview of road profile
% Inputs:
%   road_time      - time vector [s]
%   road_height    - corresponding height profile [m]
%   vehicle_speed  - [m/s]
%   preview_distance - [m], radar range ahead of car
% Output:
%   zr_future      - radar previewed profile [same size as road_time]

    % --- Calculate preview time shift
    t_shift = preview_distance / vehicle_speed;

    % --- Shift road profile in time
    % Use interpolation to simulate preview
    t_previewed = road_time + t_shift;

    % Interpolate future height values
    zr_future = interp1(road_time, road_height, t_previewed, 'linear', 'extrap');

    % Add simulated radar noise
    radar_noise = 0.005 * randn(size(zr_future));  % 5 mm noise
    zr_future = zr_future + radar_noise;
end