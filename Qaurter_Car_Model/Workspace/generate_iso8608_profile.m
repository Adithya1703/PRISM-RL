function [t, zr] = generate_iso8608_profile(road_class, v, T, fs)
% Generates ISO 8608 road profile for specified class and saves to .mat
% Inputs:
%   road_class - 'A', 'B', 'C', 'D' (case-insensitive)
%   v          - Vehicle speed in m/s (e.g., 2 for realistic)
%   T          - Duration in seconds (e.g., 10)
%   fs         - Sampling frequency in Hz (e.g., 100)
%
% Outputs:
%   t          - Time vector [s]
%   zr         - Road elevation vector [m]

    % Mapping ISO 8608 class to PSD value at reference frequency
    class_map = struct('A', 32e-6, 'B', 256e-6, 'C', 1024e-6, 'D', 8192e-6);
    road_class = upper(road_class);

    if ~isfield(class_map, road_class)
        error('Unsupported road class. Choose A, B, C, or D.');
    end

    Gd_n0 = class_map.(road_class); % PSD at reference spatial frequency
    n0 = 0.1;        % Reference spatial frequency [cycles/m]
    w = 2;           % PSD exponent (ISO 8608)
    dt = 1 / fs;
    t = 0:dt:T-dt;   % Time vector
    N = length(t);
    L = v * T;       % Total spatial road length [m]

    % Frequency domain setup
    df = 1 / L;
    f = (1:N/2) * df;                     % Spatial frequency vector
    Gd = Gd_n0 * (f / n0).^(-w);          % Target PSD
    phi = 2 * pi * rand(1, length(f));    % Random phase
    A = sqrt(2 * Gd * df);                % Amplitude spectrum

    % Build road profile using summation of harmonics
    [FF, TT] = meshgrid(f, t);            % For cosine generation
    PHI = repmat(phi, length(t), 1);
    AA = repmat(A, length(t), 1);
    z = sum(AA .* cos(2 * pi * FF .* TT + PHI), 2);  % Sum over freqs
    zr = z';  % Convert to row vector

    % Optional: Smooth transitions slightly
    zr = movmean(zr, 10);

    % Save to .mat for use in Simulink or MATLAB scripts
    filename = ['ISO8608_Class' road_class '_Profile.mat'];
    save(filename, 't', 'zr');

    % Plot
    fig = figure('Visible', 'off');
    plot(t, zr);
    xlabel('Time [s]'); ylabel('Road Elevation [m]');
    title(['ISO 8608 Class ' road_class ' Road Profile']);
    grid on;

    saveas(fig, 'road_profile.png');

end