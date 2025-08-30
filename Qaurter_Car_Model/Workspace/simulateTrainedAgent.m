%% Active Suspension: Trained RL Agent vs Passive Baseline 
% Save as: simulateTrainedAgent.m

clc; % DO NOT clear workspace; we want to preserve any loaded variables
% close all;


%% Read Sim signals
function [zs, zus, vs, vus, Ft_val, as_val] = readSimSignals(loggedSignalNames, simOut)
    % Initialize outputs
    zs = 0; 
    zus = 0; 
    vs = 0; 
    vus = 0; 
    Ft_val = 0; 
    as_val = 0;

    try
        % logs is a Simulink.SimulationData.Dataset
        logs = simOut;

        % Loop over all logged signals
        for i = 1:numel(loggedSignalNames)
            name = loggedSignalNames{i};

            try
                % Extract signal by name
                data = logs.get(name);
                if numel(data) > 1000
                    data = data(1:end-1);
                end
                lastVal = data;
            catch
                lastVal = 0;
            end

            switch name
                case 'zs',  zs     = lastVal;
                case 'zus', zus    = lastVal;
                case 'vs',  vs     = lastVal;
                case 'vus', vus    = lastVal;
                case 'Ft',  Ft_val = lastVal;
                case 'as',  as_val = lastVal;
            end
        end
    catch
        % On error, outputs remain zeros
    end
end


%% ------------------------------------------------------------------------
% 0) Configuration
% -------------------------------------------------------------------------
MODEL_NAME   = 'new_model_Qaurter_Car';   % Simulink model name
KVAR_NAME    = 'Kvar';
CVAR_NAME    = 'Cvar';
ROAD_T_NAME  = 'road_profile_time';
ROAD_H_NAME  = 'road_profile_height';

% Passive baseline parameters (factory values)
PASSIVE_K = 30000;   % N/m  
PASSIVE_C = 2500;    % Ns/m 

% Physical constants
msm_sms = 300;     % Sprung mass (kg)
mum_umu = 47;      % Unsprung mass (kg)
kt      = 200000;  % Tire stiffness (N/m)
ct      = 300;     % Tire damping (Ns/m)

% Push constants into base workspace for Simulink
assignin('base','msm_sms',msm_sms);
assignin('base','mum_umu',mum_umu);
assignin('base','kt',kt);
assignin('base','ct',ct);

%% ------------------------------------------------------------------------
% 1) Load trained agent
% -------------------------------------------------------------------------
% load('trainedSuspensionAgent.mat','agent');  % must contain variable "agent"
% disp("Loaded trained RL agent.");

%% ------------------------------------------------------------------------
% 2) Generate road profile + radar preview
% -------------------------------------------------------------------------
RoadSpeed    = 2;     % m/s
RoadDuration = 10;    % seconds
RoadFs       = 100;   % Hz
rc = 'B';             % Choose road class (A/B/C/D)

[rt, rh] = generate_iso8608_profile(rc, RoadSpeed, RoadDuration, RoadFs);
RadarPreview = simulateRadarPreview(rc, rt, rh);

% Assign road signals to workspace for Simulink
assignin('base', ROAD_T_NAME, rt(:));
assignin('base', ROAD_H_NAME, rh(:));

%% ------------------------------------------------------------------------
% 3) Run Passive Baseline Simulation
% -------------------------------------------------------------------------
assignin('base', KVAR_NAME, PASSIVE_K);
assignin('base', CVAR_NAME, PASSIVE_C);

if ~bdIsLoaded(MODEL_NAME), load_system(MODEL_NAME); end

stopTimeStr = num2str(rt(end));
simInP = Simulink.SimulationInput(MODEL_NAME);
simInP = setModelParameter(simInP, 'StopTime', stopTimeStr);
simInP = setVariable(simInP, KVAR_NAME, PASSIVE_K);
simInP = setVariable(simInP, CVAR_NAME, PASSIVE_C);
simInP = setVariable(simInP, ROAD_T_NAME, rt);
simInP = setVariable(simInP, ROAD_H_NAME, rh);

simOutPassive = sim(simInP);
loggedSignalNames = {'zs','zus','vs','vus','Ft','as'};
[zs_passive, zus_passive, ~, ~, ~, ~] = readSimSignals(loggedSignalNames, simOutPassive);

%% ------------------------------------------------------------------------
% 4) Run RL Agent Simulation
% -------------------------------------------------------------------------
% Here we simulate using the agent's chosen actions.
% Instead of simulating for each time step, we run once for the full road.
disp("Simulating with trained RL agent...");

% Preallocate
numSteps = length(rt);
k_RL   = zeros(numSteps,1);
c_RL   = zeros(numSteps,1);
zs_RL  = zeros(numSteps,1);
zus_RL = zeros(numSteps,1);

% Loop through time steps only for agent decision making
obsInfo = agent.ObservationInfo;
numObs = prod(obsInfo.Dimension);

for i = 1:numSteps
    prev_zs  = zs_RL(max(i-1,1));
    prev_zus = zus_RL(max(i-1,1));
    
    obs = zeros(numObs,1);
    obs(1) = RadarPreview(i);
    obs(2) = prev_zs;
    obs(3) = prev_zus;
    
    action = getAction(agent, obs);  
    action = action{1};  % <-- convert from cell to numeric vector
    k_val = action(1);
    c_val = action(2);
    
    k_RL(i) = k_val;
    c_RL(i) = c_val;
end


% --- Package actions as time-series for Simulink ---
K_profile = timeseries(k_RL, rt);
C_profile = timeseries(c_RL, rt);
Road_t    = timeseries(rh, rt);

% --- Setup single simulation run ---
simInRL = Simulink.SimulationInput(MODEL_NAME);
simInRL = setModelParameter(simInRL, 'StopTime', num2str(rt(end)));
simInRL = setVariable(simInRL, 'K_profile', K_profile);
simInRL = setVariable(simInRL, 'C_profile', C_profile);
simInRL = setVariable(simInRL, ROAD_T_NAME, rt);
simInRL = setVariable(simInRL, ROAD_H_NAME, rh);

% --- Run simulation once for entire road ---
simOutRL = sim(simInRL);

% --- Extract signals ---
[zs, zus, ~, ~, Ft, as] = readSimSignals(loggedSignalNames, simOutRL);

% Store results
zs_RL  = zs;
zus_RL = zus;
Ft_RL  = Ft;
as_RL  = as;


%% ------------------------------------------------------------------------
% 5) Plots
% -------------------------------------------------------------------------
figure;
subplot(2,1,1);
plot(rt, zs_RL, 'b','LineWidth',1.2); hold on;
plot(rt, zs_passive,'r--','LineWidth',1.2);
xlabel('Time (s)'); ylabel('z_s (m)');
legend('Active RL','Passive'); grid on; title('Sprung Mass Displacement');

subplot(2,1,2);
plot(rt, zus_RL, 'b','LineWidth',1.2); hold on;
plot(rt, zus_passive,'r--','LineWidth',1.2);
xlabel('Time (s)'); ylabel('z_{us} (m)');
legend('Active RL','Passive'); grid on; title('Unsprung Mass Displacement');

figure;
subplot(2,1,1);
plot(rt, Ft_RL, 'k','LineWidth',1.2);
xlabel('Time (s)'); ylabel('Force (N)'); title('Actuator Force (RL)');
grid on;
subplot(2,1,2);
plot(rt, as_RL, 'm','LineWidth',1.2);
xlabel('Time (s)'); ylabel('a_s (m/s^2)'); title('Body Acceleration (RL)');
grid on;

figure;
yyaxis left; plot(rt, k_RL,'b','LineWidth',1.1); ylabel('k (N/m)');
yyaxis right;plot(rt, c_RL,'r','LineWidth',1.1); ylabel('c (Ns/m)');
xlabel('Time (s)'); grid on; title('Agent Actions');

figure;
plot(rt, RadarPreview,'g','LineWidth',1.2);
xlabel('Time (s)'); ylabel('z_r (m)'); grid on; title('Radar Lookahead');

%% ------------------------------------------------------------------------
% 6) Metrics
% -------------------------------------------------------------------------
rms_zs_RL      = rms(zs_RL);
rms_zs_passive = rms(zs_passive);
rms_as_RL      = rms(as_RL);

fprintf('\n=== Performance ===\n');
fprintf('RMS(z_s) Active RL : %.6f m\n', rms_zs_RL);
fprintf('RMS(z_s) Passive   : %.6f m\n', rms_zs_passive);
fprintf('RMS(a_s) Active RL : %.6f m/s^2\n', rms_as_RL);
