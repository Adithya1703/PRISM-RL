% Plot the signals
figure;
plot(out.tout, out.y_s, 'b'); hold on;
plot(out.tout, out.y_u, 'r');
plot(out.tout, out.y_road, 'k--');

% Add labels and legend
legend('Sprung Mass', 'Unsprung Mass', 'Road Input');
title('Suspension System Response');
xlabel('Time (s)');
ylabel('Displacement (m)');

saveas(gcf, 'displacement.png');

% Plot accelerations
figure;
plot(out.tout, out.y_sddot, 'b', out.tout, out.y_uddot, 'r');
xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');
legend('Sprung Mass (z̈)', 'Unsprung Mass (ÿ)');
title('Accelerations of Sprung and Unsprung Mass');

saveas(gcf, 'accel.png');

% Plot suspension and tire forces
figure;
plot(out.tout, out.Fs, 'b', out.tout, out.Fd, 'g', out.tout, out.Ft, 'r');
xlabel('Time (s)');
ylabel('Force (N)');
legend('Suspension Spring Force', 'Damper Force', 'Tire Force');
title('Force Dynamics in Suspension System');

saveas(gcf, 'Forces.png');


% Plot Suspension Deflection and Tire Compression
figure;
plot(out.susp_def.Time, out.susp_def.Data);
hold on;
plot(out.tire_comp.Time, out.tire_comp.Data);
legend('Suspension Deflection(Ys-Yu)', 'Tire Compression(Yu-R)');
xlabel('Time (s)');
ylabel('Displacement (m)');
title('Suspension and Tire Dynamics');
grid on;

saveas(gcf, 'tire_comp_sus_def.png');


%* kkk = 16000;   % ks
% ccc = 1000;    % cs
% msm_sms = 290; % ms
% mum_umu = 59;  % mu
% kt = 190000;