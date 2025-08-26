classdef PredictiveSuspensionEnv < rl.env.MATLABEnvironment
    % Predictive Suspension environment for RL
    
    %% Properties(Set only once)
    properties
        % Simulation time step
        Ts = 0.01;

        % Current time Index
        CurrentStep = 1;

        StartIdx = 1;

        % Maximum Time Steps per episode( we can change this later too)
        MaxSteps = 1600;

        % Add in properties section
        EpisodeLog = []; % Will store struct array of state/action/reward per step


        % Road profile and suspension data
        RadarData          % 1xN vector : Road preview from radar
        SuspensionData     % NxS matrix [zs, zus, vs, vus, acc]

        % Action parameters
        k = 30000;                % Suspension spring constant [N/m] (action parameter)
        c = 2500;                 % Suspension damping coefficient [Ns/m] (action parameter)
        actualK = [];   % vector length N or []
        actualC = [];   % vector length N or []

        RewardWeights = [1.0, 0.8, 0.6, 0.4, 0.3, 0.05, 0.05];  % [w1, w2, w3....w7]
        CurrentEpisode = 0 % keep track of episode number

        PreviousAction = [0; 0]; % stores k & c from previous step for actuator effort penalty

        % Reward visuaization
        EpisodeRewardHistory = [];
        CurrentEpisodeReward = 0;
        maxTime        = 10;            % seconds per episode

        % Define thresholds
        Scale_a = 15;                   % m/s²    expected max sprung accel
        Scale_j = 1500;                 % m/s^3   expected jerk range (tune)
        Scale_F = 8000;                 % N       typical dynamic range
        Scale_dx = 0.10;                % m       10 cm suspension travel
        Scale_dv = 1.0;                 % m/s     rel vel scale

        % Reward scaling
        scaleR = 10;

        % Optional actuator shaping
        Scale_dk = 10000;               % N/m     spring constant change scale
        Scale_dc = 1000;                % Ns/m     damping coefficient change scale
    end

    %% Observation and Action Info
    properties(Access = protected)

        % Current observation (state)
        State = zeros(7,1);
    end

    methods
        %% Constructor
        function this = PredictiveSuspensionEnv(radar, suspension, rewardWeights)
            % Constructor: Initializes environment with radar and suspension data

            % Validate input
            if nargin < 2
                error('Constructor requires radar and suspension data.');
            end
            if numel(radar) ~= size(suspension,1)
                error('Radar and Suspension data length mismatch. Radar length must equal number of rows in suspension.');
            end

            % Observation (7)
            ObservationInfo = rlNumericSpec([7 1], ...
                'LowerLimit', [-Inf; -Inf; -Inf; -Inf; -Inf; -Inf; -Inf], ...
                'UpperLimit', [ Inf;  Inf;  Inf;  Inf;  Inf;  Inf;  Inf]);
            ObservationInfo.Name = 'states';

            % Action (k, c)
            ActionInfo = rlNumericSpec([2 1], ...
                'LowerLimit', [10000; 500], ...
                'UpperLimit', [30000; 3500]);
            ActionInfo.Name = 'actions';

            % Superclass
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);

            % Assign data
            this.RadarData = radar(:); % ensure column
            this.SuspensionData = suspension;

            % RewardWeights if provided
            if nargin >= 3 && ~isempty(rewardWeights)
                if numel(rewardWeights) ~= 7, error('rewardWeights must be length 7'); end
                this.RewardWeights = rewardWeights;
            end
        end

        %% Setter for reward weights (safe)
        function setRewardWeights(this, weights)
            if nargin < 2 || isempty(weights)
                error('Provide weights vector of length 7');
            end
            if numel(weights) ~= 7
                error('RewardWeights must be a vector of length 7.');
            end
            this.RewardWeights = weights(:).';
        end

        %% Setter for scaleR
        function setScaleR(this, newScaleR)
            validateattributes(newScaleR, {'numeric'}, {'scalar','positive'});
            this.scaleR = newScaleR;
            fprintf('scaleR updated to %.6g\n', this.scaleR);
        end

        %% Reward (cost) function -> returns POSITIVE cost (lower is better)
        function cost = rewardFunction(this, state, predicted_kc, actual_kc, params)
            % rewardFunction -> returns a POSITIVE cost (lower is better)
            % Inputs:
            %  - state: struct with fields acceleration (scalar), displacement (scalar), tire_force (scalar)
            %  - predicted_kc: [k; c] (current action)
            %  - actual_kc: [k; c] or [] (if empty, prediction error term ignored)
            %  - params: struct with alpha1..alpha5 (weights)
            %
            % Output:
            %  - cost : positive scalar (we'll convert to reward via 1 - tanh(cost/scaleR))
        
            % Defensive: ensure fields exist
            if ~isfield(state, 'acceleration'), state.acceleration = 0; end
            if ~isfield(state, 'displacement'), state.displacement = 0; end
            if ~isfield(state, 'tire_force'), state.tire_force = 0; end
        
            % squared / RMS like penalties (single-step scalar)
            rms_disp2  = (state.displacement)^2;   % single-step squared
            rms_accel2 = (state.acceleration)^2;
            rms_tire2  = (state.tire_force)^2;
        
            % Prediction error term (if actual_kc available)
            if isempty(actual_kc) || any(isnan(actual_kc))
                pred_error2 = 0;
            else
                pred_error2 = sum((predicted_kc - actual_kc).^2);
            end
        
            % Smoothness penalty: persistent previous action inside function
            persistent prev_kc
            if isempty(prev_kc)
                prev_kc = predicted_kc;
            end
            smooth_penalty = sum((predicted_kc - prev_kc).^2);
            prev_kc = predicted_kc;
        
            % Build positive cost as weighted sum (alpha should be positive)
            cost = params.alpha1 * rms_disp2 + ...
                   params.alpha2 * rms_accel2 + ...
                   params.alpha3 * rms_tire2 + ...
                   params.alpha4 * pred_error2 + ...
                   params.alpha5 * smooth_penalty;
        
            % (Optional) will add small saturation penalty if displacement normalized > 1
            % caller can add sat_pen separately if desired
        end


        %% STEP
        function [obs, reward, isDone, loggedSignals] = step(this, action)
            % STEP function: called at each time step by RL agent

            % Ensure action is within bounds
            % Note: rlNumericSpec already enforces limits, but we can double-check
            action = min(max(action, this.ActionInfo.LowerLimit), this.ActionInfo.UpperLimit);
            if numel(action) ~= 2
                error('Action must be a 2-element vector [k, c].');
            end
            action = action(:);

            % Update suspension parameters from action
            this.k = action(1);
            this.c = action(2);

            idx = this.CurrentStep; % Current data index
            epStep = idx - this.StartIdx + 1;

            % Check if the current step exceeds the length of data
            if idx > length(this.RadarData) || idx > size(this.SuspensionData,1)
                isDone = true;       % End episode if no more data is available
                obs = this.State;    % Return current state
                reward = 0;          % No reward if episode is done
                loggedSignals = [];  % No logged signals
                return;
            end


            % Get current data
            zr  = this.RadarData(idx);              % radar preview
            zs  = this.SuspensionData(idx, 1);      % Sprung pos
            zus = this.SuspensionData(idx, 2);      % unsprung pos
            vs  = this.SuspensionData(idx, 3);      % sprung vel
            vus = this.SuspensionData(idx, 4);      % unsprung vel
            Ft  = this.SuspensionData(idx, 5);      % Tire force
            as  = this.SuspensionData(idx, 6);      % sprung acceleration
            
            % Derived
            deflection = zs-zus; % Suspension deflection(spring travel)
            defl_vel = vs-vus;   % Suspension deflection velocity

            % Prepare stateStruct for rewardFunction
            stateStruct.acceleration = as;
            stateStruct.displacement = deflection;
            stateStruct.tire_force = Ft;

            % Jerk(derivative of acceleration)
            if idx>1
                as_prev = this.SuspensionData(idx-1, 6); % Previous acceleration
                jerk = (as - as_prev) / this.Ts; % Jerk calculation
            else
                jerk = 0; % No previous data for first step
            end

            % Actuator effort
            if idx>1
                delta_k = action(1) - this.PreviousAction(1); % Change in spring constant
                delta_c = action(2) - this.PreviousAction(2); % Change in damping coefficient
            else
                delta_c = 0; % No previous action for first step
                delta_k = 0;
            end

            % actual_kc if available
            if ~isempty(this.actualK) && ~isempty(this.actualC) && length(this.actualK) >= idx && length(this.actualC) >= idx
                actual_kc = [ this.actualK(idx); this.actualC(idx) ];
            else
                actual_kc = [];  % gracefully ignore prediction-error term
            end
            predicted_kc = action(:);

            % Define reward weights as a struct with alpha fields or vector
            % as before(tunable)
            params.alpha1 = 10;  % displacement penalty weight
            params.alpha2 = 10;  % acceleration penalty weight
            params.alpha3 = 5;   % tire force penalty weight
            params.alpha4 = 100; % prediction error penalty weight
            params.alpha5 = 1;   % smoothness penalty weight

            % Calculate reward based on the updated state
            % dummy reward we can use -> reward = -abs(zs);
            % reward = -norm(this.State(1:2));  % Example reward function

            % Reward function: Penalize large displacement and velocity (comfort & stability)
            % reward = -abs(zs) - 0.1*abs(vs);

            % Normalized quantities
            na = as/this.Scale_a;
            nj = jerk/this.Scale_j;
            nF = Ft/this.Scale_F;
            ndx = deflection/this.Scale_dx;
            ndv = defl_vel/this.Scale_dv;
            ndk = delta_k/this.Scale_dk;
            ndc = delta_c/this.Scale_dc;

            % --- Soft saturation penalty if travel nears/exceeds 1*Scale_dx ---
            % zero inside |ndx|<=1, quadratic growth outside
            sat = 2.0; % saturation - physically critical
            % In suspension control, saturation means when the suspension displacement (ndx) exceeds the physical stroke limit (like the damper hitting the bump stop or topping out).
            % Example: If your normalized suspension travel (ndx) is supposed to stay in [-1, 1], then anything beyond ±1 is a violation.
            sat_pen = max(0, abs(ndx) - 1).^2;

            % Define reward weights
            w = this.RewardWeights;

            if numel(w) ~= 7
                error('RewardWeights must be a 7-element vector');
            end

            % % reward calculation
            % cost = this.rewardFunction(stateStruct, predicted_kc, actual_kc, params);
            cost = ...
                w(1) * (na)^2 ...
                + w(2) * (nj)^2 ...
                + w(3) * (nF)^2 ...
                + w(4) * (ndv)^2 ... 
                + w(5) * (ndx)^2 ...
                + sat * sat_pen ...
                + w(6) * (ndk)^2 ...
                + w(7) * (ndc)^2;
            % 
            % % reward = max(-1000, min(0, -cost)); % Negative cost as reward
            reward = 1 - tanh(cost/this.scaleR);
            % reward = this.rewardFunction(stateStruct, predicted_kc, actual_kc, params);

            % small bonus when inside all hard safety envelopes
            if abs(ndx) <= 1 && abs(nF) <= 1 && abs(na) <= 1
                reward = reward + 0.02;       % gentle shaping bonus
            end
            % reward = min(reward, 1.0);        % keep within (0,1]
            % Clip reward to [-1,1] to avoid runaway values
            reward = max(min(reward, 1), -1);

            this.CurrentEpisodeReward = this.CurrentEpisodeReward + reward; % Accumulate reward for the episode
            
            % Update state vector
            this.State = [ ...
                        zr; ...
                        zs/this.Scale_dx; ...
                        zus/this.Scale_dx; ...
                        vs/this.Scale_dv; ...
                        vus/this.Scale_dv; ...
                        Ft/this.Scale_F; ...
                        as/this.Scale_a ];
            
            
            % Store previous action for next step
            this.PreviousAction = action(:);

            % debug check
            randIdx = randi([1, length(this.RadarData)]);
            disp([this.SuspensionData(randIdx,6), this.SuspensionData(randIdx,5)]);


            % Check if the episode is done
            elapsedTime = (this.CurrentStep - this.StartIdx) * this.Ts;
            isDone = (epStep >= this.MaxSteps) ...
                  || (this.CurrentStep > length(this.RadarData)) ...
                  || (this.CurrentStep > size(this.SuspensionData,1)) ...
                  || (elapsedTime >= this.maxTime);
                  %|| (abs(as) > this.Scale_a) ...
                  %|| (abs(Ft) > this.Scale_F);

            % Reward accumulation
            if isDone
                disp(['Terminated: epStep=' num2str(epStep) ...
                      ', CurrentStep=' num2str(this.CurrentStep) ...
                      ', idx=' num2str(idx) ...
                      ', as=' num2str(as) ...
                      ', Ft=' num2str(Ft) ...
                      ', this.MaxSteps=' num2str(this.MaxSteps) ...
                      ', this.maxTime=' num2str(this.maxTime) ...
                      ', (CurrentStep*Ts)=' num2str(this.CurrentStep*this.Ts)]);

            end

            % Advance step
            this.CurrentStep = this.CurrentStep + 1;
            
            % Prepare logged signals (if any)
            loggedSignals = struct('State', this.State, 'Action', action, 'Cost', cost, 'Reward', reward);
            % loggedSignals = [];

            % Append to episode log
            % Store the current state, action, and reward in the episode log
            logIndex = max(1, min(epStep, this.MaxSteps));
            this.EpisodeLog(logIndex) = struct('State', this.State, 'Action', action, 'Reward', reward);

            % Return the observation
            obs = this.State;

        end

        %% getActionBounds
        function [kBounds, cBounds] = getActionBounds(this)
            kBounds = [this.ActionInfo.LowerLimit(1), this.ActionInfo.UpperLimit(1)];
            cBounds = [this.ActionInfo.LowerLimit(2), this.ActionInfo.UpperLimit(2)];
        end

        %% plotEpisode
        function plotEpisode(this)
            log = this.getValidEpisodeLog();
            if isempty(log), warning('No episode log available'); return; end
            zs_vals = arrayfun(@(x)x.State(2), log);
            vs_vals = arrayfun(@(x)x.State(4), log);

            figure;
            plot(zs_vals, 'DisplayName', 'zs (agent)');
            hold on;
            plot(vs_vals, 'DisplayName', 'vs (agent)');
            legend;
            title('Agent-Controlled Sprung Mass Position and Velocity');
            xlabel('Time Step');
            ylabel('Value');
            grid on;
        end

        %% plotRewardCurve
        function plotRewardCurve(this, window)
            if nargin < 2, window = 5; end
            rewards = this.EpisodeRewardHistory;
            if isempty(rewards), disp('No reward history found.'); return; end
            if length(rewards) >= window
                movingAvg = movmean(rewards, window);
            else
                movingAvg = rewards;
            end
            figure;
            plot(rewards, 'b-', 'DisplayName', 'Raw Reward');
            hold on;
            plot(movingAvg, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('Moving Avg (window=%d)', window));
            xlabel('Episode'); ylabel('Total Reward');
            title('Reward Curve Over Episodes'); legend; grid on;
        end

        %% getState
        function s = getState(this)
            s = this.State;
        end
        
        %% Reset
        function reset(this)
            % Initialize or reset episode log
            this.EpisodeLog = repmat(struct('State', [], 'Action', [], 'Reward', []), this.MaxSteps, 1);
        
            % Randomize initial action parameters
            this.k = 18000 + 4000 * rand;
            this.c = 2000 + 1000 * rand;
        
            % Log previous episode's reward if it exists
            if ~isempty(this.CurrentEpisodeReward)
                this.EpisodeRewardHistory(end + 1) = this.CurrentEpisodeReward;
            end
        
            % Reset episode reward and increment episode count
            this.CurrentEpisodeReward = 0;
            this.CurrentEpisode = this.CurrentEpisode + 1;
        
            % Print average reward every 10 episodes
            if this.CurrentEpisode >= 10 && mod(this.CurrentEpisode, 10) == 0
                avgReward = mean(this.EpisodeRewardHistory(end-9:end));
                fprintf('Average Reward (Episodes %d–%d): %.2f\n', this.CurrentEpisode - 9, this.CurrentEpisode, avgReward);
            end
            
            % Store previous action for effort penalty
            this.PreviousAction = [this.k; this.c];
        
            % Data check
            if isempty(this.RadarData) || isempty(this.SuspensionData)
                error('Radar and Suspension data must be provided before reset.');
            end
        
            % % Sequential episode start index to cover all data
            % if isempty(this.CurrentStep) || this.CurrentStep + this.MaxSteps - 1 > length(this.RadarData)
            %     this.CurrentStep = 1;  % restart from beginning if overflow
            % end
            % this.StartIdx = this.CurrentStep;

            % Choose a random valid StartIdx so an episode of MaxSteps fits
            maxStartIdx = length(this.RadarData) - this.MaxSteps + 1;
            if maxStartIdx < 1
                error('Not enough data for an episode with MaxSteps.');
            end
            this.StartIdx = randi([1, maxStartIdx]);
        
            % Set CurrentStep to StartIdx (begin at start of episode)
            this.CurrentStep = this.StartIdx;
        
            % Initialize normalized state at StartIdx (consistent with step())
            zr  = this.RadarData(this.StartIdx);                    % raw radar preview
            zs  = this.SuspensionData(this.StartIdx, 1) / this.Scale_dx;
            zus = this.SuspensionData(this.StartIdx, 2) / this.Scale_dx;
            vs  = this.SuspensionData(this.StartIdx, 3) / this.Scale_dv;
            vus = this.SuspensionData(this.StartIdx, 4) / this.Scale_dv;
            Ft  = this.SuspensionData(this.StartIdx, 5) / this.Scale_F;
            as  = this.SuspensionData(this.StartIdx, 6) / this.Scale_a;
        
            this.State = [zr; zs; zus; vs; vus; Ft; as];
        
            % Print episode start info
            fprintf('Episode %d started. Start index: %d Initial state: [%0.3f %0.3f ...]\n', ...
                    this.CurrentEpisode, this.StartIdx, this.State(1), this.State(2));
        
            % Advance current step to next episode start index
            this.CurrentStep = this.StartIdx;
        end


        %% getVaidEpsiodeLog
        function validLog = getValidEpisodeLog(this)
            % Returns only the filled portion of the episode log
            maxLogStep = min(this.MaxSteps, find(arrayfun(@(x)~isempty(x.State), this.EpisodeLog), 1, 'last'));
            if isempty(maxLogStep)
                validLog = [];
            else
                validLog = this.EpisodeLog(1:maxLogStep);
            end
        end
    end
end


