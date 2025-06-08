% ISO 8608 Road Profile Generator - Class B (Corrected)
clc; clear;

% Parameters
Gd_n0 = 256e-6;      % Class B: PSD at reference spatial frequency
n0 = 0.1;            % Reference spatial frequency [cycles/m]
w = 2;               % Exponent in PSD function
v = 2;              % Vehicle speed [m/s]
T = 10;              % Duration [s]
fs = 100;            % Sampling frequency [Hz]
dt = 1/fs;
road_profile_time = 0:dt:T-dt;       % Time vector
N = length(road_profile_time);
L = v * T;           % Total length of road [m]

% Spatial frequency vector (exclude DC)
df = 1/L;
f = (1:N/2) * df;    % Half spectrum (positive frequencies)
Gd = Gd_n0 * (f / n0).^(-w);  % PSD vector

% Random phase for each frequency
phi = 2 * pi * rand(1, length(f));

% Amplitude spectrum from PSD
A = sqrt(2 * Gd * df);  % Factor 2 for real signal symmetry

% Generate road profile using summation of harmonics
% Broadcasting-safe: f is [1 x M], t is [1 x N]
% So we use meshgrid to ensure dimensions align
[FF, TT] = meshgrid(f, road_profile_time);       % FF: [N x M], TT: [N x M]
PHI = repmat(phi, length(road_profile_time), 1); % [N x M]
AA = repmat(A, length(road_profile_time), 1);    % [N x M]

% Sum across frequencies
z = sum(AA .* cos(2 * pi * FF .* TT + PHI), 2);  % [N x 1]
zr = z';  % Convert to row vector for consistency

% Optional: Smooth transitions slightly
road_profile_height = movmean(zr, 10);  

% Plot
plot(road_profile_time, road_profile_height);
xlabel('Time [s]');
ylabel('Road Elevation [m]');
title('ISO 8608 Class B Road Profile');
grid on;

% Save to .mat
save('ISO8608_ClassB_Profile.mat', 'road_profile_time', 'road_profile_height');
saveas(gcf, 'road_profile.png');