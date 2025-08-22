% Plot the signals
fig = figure('Visible', 'off');
plot(out.tout, out.y_s, 'b'); hold on;
plot(out.tout, out.y_u, 'r');
plot(out.tout, out.y_road, 'k--');

% Add labels and legend
legend('Sprung Mass', 'Unsprung Mass', 'Road Input');
title('Suspension System Response');
xlabel('Time (s)');
ylabel('Displacement (m)');

saveas(fig, 'displacement.png');

% Plot accelerations
fig = figure('Visible', 'off');
plot(out.tout, out.y_sddot, 'b', out.tout, out.y_uddot, 'r');
xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');
legend('Sprung Mass (z̈)', 'Unsprung Mass (ÿ)');
title('Accelerations of Sprung and Unsprung Mass');

saveas(fig, 'accel.png');

% Plot suspension and tire forces
fig = figure('Visible', 'off');
plot(out.tout, out.Fs, 'b', out.tout, out.Fd, 'g', out.tout, out.Ft, 'r');
xlabel('Time (s)');
ylabel('Force (N)');
legend('Suspension Spring Force', 'Damper Force', 'Tire Force');
title('Force Dynamics in Suspension System');

saveas(fig, 'Forces.png');


% Plot Suspension Deflection and Tire Compression
fig = figure('Visible', 'off');
plot(out.susp_def.Time, out.susp_def.Data);
hold on;
plot(out.tire_comp.Time, out.tire_comp.Data);
legend('Suspension Deflection(Ys-Yu)', 'Tire Compression(Yu-R)');
xlabel('Time (s)');
ylabel('Displacement (m)');
title('Suspension and Tire Dynamics');
grid on;
saveas(fig, 'tire_comp_sus_def.png');


% Open file
fid = fopen('acceleration_data.csv', 'w');

% Write headers
fprintf(fid, 'Time (s),Sprung Accel (m/s^2),Unsprung Accel (m/s^2), Spring Force (N), Damper Force (N), Tire force(N)\n');

% Write data row by row
for i = 1:length(out.tout)
    fprintf(fid, '%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', out.tout(i), out.y_sddot(i), out.y_uddot(i), out.Fs(i), out.Fd(i), out.Ft(i));
end

fclose(fid);
%* kkk = 16000;   % ks
% ccc = 1000;    % cs
% msm_sms = 290; % ms
% mum_umu = 59;  % mu
% kt = 190000;