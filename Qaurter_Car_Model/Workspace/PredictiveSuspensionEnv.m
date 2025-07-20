classdef PredictiveSuspensionEnv < rl.env.MATLABEnvironment
    % Predictive Suspension environment for RL
    
    %% Properties(Set only once)
    properties
        % Simulation time step
        Ts = 0.01;

        % Current time Index
        CurrentStep = 1;

        % Maximum Time Steps per episode( we can change this later too)
        MaxSteps = 1000;

        % Add in properties section
        EpisodeLog = []; % Will store struct array of state/action/reward per step


        % Road profile and suspension data
        RadarData          % 1xN vector : : Road preview from radar
        SuspensionData     % NxS matrix [zs, zus, vs, vus, acc]

        % Action parameters
        k = 20000;                % Suspension spring constant [N/m] (action parameter)
        c = 2500;                 % Suspension damping coefficient [Ns/m] (action parameter)

        RewardWeights = [1.0, 0.8, 0.3, 0.1, 0.05]; % [w1, w2, w3, w4, w5]

        UnsprungMass = 40; % Unsprung mass [kg] (constant)
        PreviousAction = [0; 0]; % stores k & c from previous step for actuator effort penalty
    end

    %% Observation and Action Info
    properties(Access = protected)
        % Current observation (state)
        State = zeros(5,1);
    end

    methods
        function this = PredictiveSuspensionEnv(radar, suspension, rewardWeights)
            % Constructor: Initializes environment with radar and suspension data

            if nargin > 2
                this.RewardWeights = rewardWeights;
            end

            % Validate input dimensions
            if numel(radar) ~= size(suspension,1)
                error('Radar and Suspension data length mismatch. Radar must be 1xN, Suspension must be NxS.');
            end

            % Define observation (5 state values): [zr, zs, zus, vs, vus]
            ObservationInfo = rlNumericSpec([5 1], ...
                'LowerLimit', [-2;-Inf;-Inf;-Inf;-Inf], ...
                'UpperLimit', [2;Inf;Inf;Inf;Inf]);
            ObservationInfo.Name = 'states';

            % Define action (2 values: k and c)
            ActionInfo = rlNumericSpec([2 1], ...
                'LowerLimit', [10000; 500], ...
                'UpperLimit', [30000; 3500]);
            ActionInfo.Name = 'actions';
            
            % Call the superclass constructor
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);

            % Assign radar and suspension input data to properties
            this.RadarData = radar;
            this.SuspensionData = suspension;

            this.ObservationInfo = ObservationInfo;  % to access outside the constructor
            this.ActionInfo = ActionInfo;            % to access outside the constructor
        end

        function setRewardWeights(this, weights)
            if numel(weights) ~= 5
                error('RewardWeights must be a vector of length 5.');
            end
            this.RewardWeights = weights;
        end

        function [obs, reward, isDOne, loggedSignals] = step(this, action)
            % STEP function: called at each time step by RL agent

            % Ensure action is within bounds
            % Note: rlNumericSpec already enforces limits, but we can double-check
            action = min(max(action, this.ActionInfo.LowerLimit), this.ActionInfo.UpperLimit);


            % Update suspension parameters from action
            this.k = action(1);
            this.c = action(2);

            idx = this.CurrentStep; % Current data index

            % Check if the current step exceeds the length of data
            if idx > length(this.RadarData) || idx > size(this.SuspensionData,1)
                isDOne = true;       % End episode if no more data is available
                obs = this.State;    % Return current state
                reward = 0;          % No reward if episode is done
                loggedSignals = [];  % No logged signals
                return;
            end

            % Get current data
            zr = this.RadarData(idx);              % radar preview
            zs = this.SuspensionData(idx, 1);      % Sprung pos
            zus = this.SuspensionData(idx, 2);     % unsprung pos
            vs = this.SuspensionData(idx, 3);      % sprung vel
            vus = this.SuspensionData(idx, 4);     %unsprung vel
            Ft = this.SuspensionData(idx, 5);      % Tire force
            as = this.SuspensionData(idx, 6);      % sprung acceleration

            % this.State = this.SuspensionData(this.CurrentStep, :);
            
            % Suspension deflection(spring travel)
            deflection = zs-zus;

            % Tire force deviation from nominal (optional)
            Ft_nom = this.UnsprungMass*9.81; % Nominal tire force (mg)

            % Jerk(derivative of acceleration)
            if idx>1
                as_prev = this.SuspensionData(idx-1, 6); % Previous acceleration
                jerk = (as - as_prev) / this.Ts; % Jerk calculation
            else
                jerk = 0; % No previous data for first step
            end

            % Actuator effort
            if idx>1
                prevAction = this.PreviousAction;
                delta_c = action(2) - prevAction(2); % Change in damping coefficient
                delta_k = action(1) - prevAction(1); % Change in spring constant
            else
                delta_c = 0; % No previous action for first step
                delta_k = 0;
            end

            % Calculate reward based on the updated state
            % dummy reward we can use -> reward = -abs(zs);
            % reward = -norm(this.State(1:2));  % Example reward function

            % Reward function: Penalize large displacement and velocity (comfort & stability)
            % reward = -abs(zs) - 0.1*abs(vs);
            %reward = - this.RewardWeights(1) * abs(zs) - this.RewardWeights(2) * abs(vs);


            % Define reward weights
            w = this.RewardWeights;

            % Reward calculation
            reward = ...
                -w(1) * (as)^2 ...               % Penalize acceleration
                -w(2) * (Ft - Ft_nom)^2 ...      % Penalize tire force deviation
                -w(3) * (deflection)^2 ...       % Penalize suspension travel
                -w(4) * (jerk)^2 ...              % Penalize jerk
                -w(5) * (delta_c^2 + delta_k^2); % Penalize actuator effort
            
            % Update state vector
            this.State = [zr; zs; zus; vs; vus];   % Update state vector
            
            % Store previous action for next step
            this.PreviousAction = action;


            % Check if the episode is done
            isDOne = this.CurrentStep >= this.MaxSteps;

            % Advance step
            this.CurrentStep = this.CurrentStep + 1;
            
            % Prepare logged signals (if any)
            loggedSignals = struct('State', this.State, 'Action', action);
            % loggedSignals = [];

            % Append to episode log
            % Store the current state, action, and reward in the episode log
            this.EpisodeLog(idx) = struct('State', this.State, 'Action', action, 'Reward', reward);

            % Return the observation
            obs = this.State;

        end

        function [kBounds, cBounds] = getActionBounds(this)
            kBounds = [this.ActionInfo.LowerLimit(1), this.ActionInfo.UpperLimit(1)];
            cBounds = [this.ActionInfo.LowerLimit(2), this.ActionInfo.UpperLimit(2)];
        end

        % method to plot episode trajectory
        function plotEpisode(this)
            log = this.getValidEpisodeLog();
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

        function reset(this)
            % reset function for environment
            this.EpisodeLog = repmat(struct('State', [], 'Action', [], 'Reward', []), this.MaxSteps, 1);
            this.PreviousAction = [this.k; this.c];
            this.CurrentStep = 1;

            % Randomize initial state for generalization
            this.State = [ ...
                this.RadarData(1); ...
                this.SuspensionData(1, 1) + 0.01*randn; ... % zs with small noise
                this.SuspensionData(1, 2) + 0.01*randn; ... % zus with small noise
                this.SuspensionData(1, 3) + 0.01*randn; ... % vs with small noise
                this.SuspensionData(1, 4) + 0.01*randn];    % vus with small noise

            % Optionally, allow user to specify initial state as an argument
        end

        function validLog = getValidEpisodeLog(this)
            % Returns only the filled portion of the episode log
            validLog = this.EpisodeLog(1:this.CurrentStep-1);
        end
    end
end


