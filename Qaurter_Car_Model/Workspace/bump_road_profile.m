% Road Profile - Bump followed by a Pothole

% Section 1: Flat road before bump
x_flat1 = 0:0.1:1.1;
z_flat1 = zeros(1, length(x_flat1));

% Section 2: Bump (semi-circle up)
R_bump = 0.20; % radius of bump
th_bump = 0:0.01:pi;
x_bump = -R_bump*cos(th_bump) + x_flat1(end) + R_bump;
z_bump = R_bump*sin(th_bump);

% Section 3: Flat road between bump and pothole
x_flat2 = x_bump(end):0.1:x_bump(end) + 1;
z_flat2 = zeros(1, length(x_flat2));

% Section 4: Pothole (semi-circle down)
R_pothole = 0.15; % radius of pothole
th_pothole = 0:0.01:pi;
x_pothole = -R_pothole*cos(th_pothole) + x_flat2(end) + R_pothole;
z_pothole = -R_pothole*sin(th_pothole); % negative dip

% Section 5: Flat road after pothole
x_flat3 = x_pothole(end):0.1:x_pothole(end) + 5;
z_flat3 = zeros(1, length(x_flat3));

% Combine all sections
X_r = [x_flat1, x_bump(2:end), x_flat2(2:end), x_pothole(2:end), x_flat3(2:end)];
Z_r = [z_flat1, z_bump(2:end), z_flat2(2:end), z_pothole(2:end), z_flat3(2:end)];

% Convert distance to time (assuming constant speed)
v = 2; % [m/s]
road_profile_time = X_r / v;
road_profile_height = Z_r;

% Push to base workspace
assignin('base', 'road_profile_time', road_profile_time);
assignin('base', 'road_profile_height', road_profile_height);


plot(X_r, Z_r, 'LineWidth', 2)
xlabel('Distance [m]')
ylabel('Road Profile Height [m]')
title('Road Profile with Bump and Pothole')
grid on;

% Set y-axis ticks from 0 to 10 in steps of 1
yticks(-2:1:8)
ylim([-2 8])

saveas(gcf, 'road_profile.png');