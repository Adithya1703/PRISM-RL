classdef PredictiveSuspensionEnv < rl.env.MATLABEnvironment
    % PredictiveSuspensionEnv - Refactored
    % See comments in code for important changes.

    properties
        % Timing & limits
        Ts = 0.01;
        MaxSteps = 1600;
        maxTime = 10;

        % Episode bookkeeping
        CurrentStep = 1;
        StartIdx = 1;
        CurrentEpisode = 0;
        EpisodeLog = [];           % struct array preallocated at reset
        EpisodeRewardHistory = [];
        CurrentEpisodeReward = 0;

        % Optional precomputed data
        RadarData = [];
        SuspensionData = [];
        PrecomputedSuspension = [];

        % Action/state
        k = 30000;
        c = 2500;
        PreviousAction = [0;0];

        % Simulink scalar parameters (ensure exist in base)
        msm_sms = 300;
        mum_umu = 47;
        kt = 200000;
        ct = 300;

        % Optional actual k/c sequences (prediction-error term)
        actualK = [];
        actualC = [];

        % Reward weighting
        RewardWeights = [0.40, 0.15, 0.15, 0.08, 0.12, 0.03, 0.02];

        % Scaling / normalization
        Scale_a = 15;
        Scale_j = 1500;
        Scale_F = 8000;
        Scale_dx = 0.10;
        Scale_dv = 1.0;
        Scale_dk = 10000;
        Scale_dc = 1000;
        scaleR = 10;    % MUST be > 0; createEnvironment will try to set sensibly

        % Simulink names (adapt to your model)
        ModelName = 'new_model_Qaurter_Car';    % change to your model name
        KVarName = 'Kvar';
        CVarName = 'Cvar';
        RoadTimeVar = 'road_profile_time';
        RoadHeightVar = 'road_profile_height';

        % Road generation defaults (used by reset())
        RoadClasses = {'A','B','C','D'};
        RoadSpeed = 2;
        RoadDuration = 10;
        RoadFs = 100;

        % Expected logsout names
        LoggedSignalNames = {'zs','zus','vs','vus','Ft','as'};
    end

    properties (Access = protected)
        State = zeros(7,1);   % [ zr; zs; zus; vs; vus; Ft; as ] normalized except zr
        RadarPreview = [];
        RoadTime = [];
        RoadHeight = [];
        SimTime = 0;          % monotonic stop time used to advance sim
        SimulinkAvailable = false;
        UseBaseWorkspace = false; % we prefer SimulationInput but keep base fallback
    end

    methods
        %% Constructor
        function this = PredictiveSuspensionEnv(varargin)
            obsInfo = rlNumericSpec([7 1], 'LowerLimit', -inf*ones(7,1), 'UpperLimit', inf*ones(7,1));
            obsInfo.Name = 'states';
            % Action limits: keep these consistent with actor scaling in runTraining.m
            actLB = [12000; 900];
            actUB = [35000; 5000];
            actInfo = rlNumericSpec([2 1], 'LowerLimit', actLB, 'UpperLimit', actUB);
            actInfo.Name = 'actions';
            this = this@rl.env.MATLABEnvironment(obsInfo, actInfo);

            % Optionally accept precomputed datasets and reward weights
            if nargin >= 1 && ~isempty(varargin{1}), this.RadarData = varargin{1}(:); end
            if nargin >= 2 && ~isempty(varargin{2}), this.SuspensionData = varargin{2}; end
            if nargin >= 3 && ~isempty(varargin{3})
                if numel(varargin{3}) ~= 7, error('RewardWeights must be length 7'); end
                this.RewardWeights = varargin{3}(:).';
            end
            this.ensureWorkspaceParameters();

            % Preallocate episode log
            this.EpisodeLog = repmat(struct('State', [], 'Action', [], 'Reward', []), this.MaxSteps, 1);

            % Attempt to load Simulink model and enable FastRestart if possible
            try
                load_system(this.ModelName);
                % Turn FastRestart on if supported
                try
                    set_param(this.ModelName, 'FastRestart', 'off');
                catch
                    % older versions may not support; ignore
                end
                this.SimulinkAvailable = true;
                this.UseBaseWorkspace = true;
                % Optionally validate that logged signal names exist by running a very short sim
                try
                    this.validateLoggedSignals(); % non-fatal
                catch
                    % leave as-is; validation is optional
                end
            catch ME
                warning('Could not load model "%s": %s', this.ModelName, getReport(ME,'basic'));
                this.SimulinkAvailable = false;
                this.UseBaseWorkspace = false;
            end
        end

        %% Reset - start new episode
        function initialObs = reset(this)
            % Append previous episode reward
            this.CurrentEpisode = this.CurrentEpisode + 1;
            if ~isempty(this.CurrentEpisodeReward)
                this.EpisodeRewardHistory(end+1) = this.CurrentEpisodeReward;
            end
            this.CurrentEpisodeReward = 0;
        
            % Reset step
            this.CurrentStep = 1;
            this.StartIdx = 1;
            this.SimTime = 0;

            % Initialize actuators            
            this.k = 18000 + 4000*rand;
            this.c = 2000 + 1000*rand;
            this.PreviousAction = [this.k; this.c];
        
            % Randomize road & radar
            rc = this.RoadClasses{ randi(numel(this.RoadClasses)) };
            [rt, rh] = generate_iso8608_profile(rc, this.RoadSpeed, this.RoadDuration, round(this.RoadFs));
            this.RoadTime = rt(:)';
            this.RoadHeight = rh(:)';
            this.RadarPreview = simulateRadarPreview(rc, this.RoadTime, this.RoadHeight);
            nT = length(this.RadarPreview);

            this.ensureWorkspaceParameters();
        
            % Push to base workspace if needed
            if this.UseBaseWorkspace
                assignin('base', this.KVarName, this.k);
                assignin('base', this.CVarName, this.c);
                assignin('base', this.RoadTimeVar, this.RoadTime);
                assignin('base', this.RoadHeightVar, this.RoadHeight);
            end
        
            % Preallocate episode log
            this.EpisodeLog = repmat(struct('State', [], 'Action', [], 'Reward', []), this.MaxSteps, 1);
        
            % Optional: run one short sim to fill PrecomputedSuspension
            if this.SimulinkAvailable
                try
                    simIn = Simulink.SimulationInput(this.ModelName);
                    simIn = simIn.setModelParameter('StopTime', num2str(this.RoadTime(end)));
                    simIn = setVariable(simIn, this.KVarName, this.k);
                    simIn = setVariable(simIn, this.CVarName, this.c);
                    simIn = setVariable(simIn, this.RoadTimeVar, this.RoadTime);
                    simIn = setVariable(simIn, this.RoadHeightVar, this.RoadHeight);
                    % also ensure scalar params visible to model run
                    simIn = setVariable(simIn, 'msm_sms', this.msm_sms);
                    simIn = setVariable(simIn, 'mum_umu', this.mum_umu);
                    simIn = setVariable(simIn, 'kt', this.kt);
                    simIn = setVariable(simIn, 'ct', this.ct);
                    
                    %% --- Debug instrumentation (insert into reset() around sim call) ---
                    % Print sizes of variables you pass to the model
                    % varsToCheck = { this.KVarName, this.CVarName, this.RoadTimeVar, this.RoadHeightVar };
                    % fprintf('--- Pre-sim variables in MATLAB workspace ---\n');
                    % for i=1:numel(varsToCheck)
                    %     nm = varsToCheck{i};
                    %     try
                    %         v = evalin('base', nm);
                    %         fprintf('%s : size=%s class=%s\n', nm, mat2str(size(v)), class(v));
                    %     catch
                    %         fprintf('%s : <not in base workspace>\n', nm);
                    %     end
                    % end
                    % fprintf('this.RadarPreview size: %s\n', mat2str(size(this.RadarPreview)));
                    % fprintf('this.PrecomputedSuspension size: %s\n', mat2str(size(this.PrecomputedSuspension)));
                    
                    % Run sim and capture simOut
                    simOut = sim(simIn);
                    
                    % Inspect simOut top-level contents
                    % fprintf('--- simOut top-level properties ---\n');
                    % props = properties(simOut);
                    % for i=1:numel(props), fprintf('%s\n', props{i}); end
                    % 
                    % % If logsout exists, inspect each element's data shape
                    % if isprop(simOut,'logsout') && ~isempty(simOut.logsout)
                    %     ds = simOut.logsout;
                    %     fprintf('logsout: class=%s\n', class(ds));
                    %     % Attempt to get number of elements robustly
                    %     try nElements = ds.numElements; catch; nElements = ds.getNumElements(); end
                    %     for j = 1:nElements
                    %         try
                    %             entry = ds.get(j); % get by index
                    %             name = entry.Name;
                    %             if isprop(entry,'Values')
                    %                 try
                    %                     data = entry.Values.Data;
                    %                     fprintf('Signal "%s": Data size = %s, class=%s\n', name, mat2str(size(data)), class(data));
                    %                 catch ME
                    %                     fprintf('Signal "%s": couldn''t read Values.Data: %s\n', name, ME.message);
                    %                 end
                    %             else
                    %                 fprintf('Signal "%s": no Values property, class=%s\n', name, class(entry));
                    %             end
                    %         catch ME
                    %             fprintf('Could not get logsout element %d: %s\n', j, ME.message);
                    %         end
                    %     end
                    % else
                    %     fprintf('simOut has no logsout or logsout empty.\n');
                    % end
                    % simOut = sim(simIn);
        
                    % Extract all signals
                    [zs, zus, vs, vus, Ft_val, as_val] = readSimSignals(this, simOut);

                    % Store as Nx6 matrix
                    this.PrecomputedSuspension = [zs(:), zus(:), vs(:), vus(:), Ft_val(:), as_val(:)];
                catch ME
                    % Fallback to zeros if simulation fails
                    warning('Pre-sim in reset failed; falling back to zeros: %s', getReport(ME,'basic'));
                    this.PrecomputedSuspension = zeros(nT,6);
                end
            else
                % if no Simulink available, fallback to zeros or existing SuspensionData if provided
                if ~isempty(this.SuspensionData)
                    L = min(size(this.SuspensionData,1), nT);
                    tmp = zeros(nT,6);
                    tmp(1:L,:) = this.SuspensionData(1:L,1:6);
                    this.PrecomputedSuspension = tmp;
                else
                    this.PrecomputedSuspension = zeros(nT,6);
                end
            end
        
            % Initial observation (radar raw, rest normalized)
            zr0 = this.safeIndex(this.RadarPreview, 1);
            this.State = [ zr0; zeros(6,1) ];
            initialObs = this.State;
        end


        %% ensureWorkspaceParameters
        function ensureWorkspaceParameters(this)
            % Always set required scalar parameters in base workspace
            assignin('base','msm_sms', this.msm_sms);
            assignin('base','mum_umu', this.mum_umu);
            assignin('base','kt', this.kt);
            assignin('base','ct', this.ct);
        end


        %% Step Function - will not run simulink model
        function [obs, reward, isDone, loggedSignals] = step(this, action)
            % Clip action
            rawAction = action(:);
            action = min(max(rawAction, this.ActionInfo.LowerLimit), this.ActionInfo.UpperLimit);

            % Update actuator variables
            this.k = action(1); this.c = action(2);
        
            % Update base workspace if required
            if this.UseBaseWorkspace
                assignin('base', this.KVarName, this.k);
                assignin('base', this.CVarName, this.c);
                this.ensureWorkspaceParameters();
            end
        
            idx = this.CurrentStep;
        
            % End episode if out of data
            if idx > length(this.RadarPreview)
                isDone = true;
                obs = this.State;
                reward = 0;
                loggedSignals = struct();
                return;
            end
        
            % Extract simulation signals for current step
            if ~isempty(this.PrecomputedSuspension) && size(this.PrecomputedSuspension,1) >= idx
                zs  = this.PrecomputedSuspension(idx,1);
                zus = this.PrecomputedSuspension(idx,2);
                vs  = this.PrecomputedSuspension(idx,3);
                vus = this.PrecomputedSuspension(idx,4);
                Ft_val = this.PrecomputedSuspension(idx,5);
                as_val = this.PrecomputedSuspension(idx,6);
            else
                % As last resort attempt a one-step sim run (rare)
                if this.SimulinkAvailable
                    try
                        stopT = this.SimTime + this.Ts;
                        simIn = Simulink.SimulationInput(this.ModelName);
                        simIn = setModelParameter(simIn, 'StopTime', num2str(stopT));
                        simIn = setVariable(simIn, this.KVarName, this.k);
                        simIn = setVariable(simIn, this.CVarName, this.c);
                        simIn = setVariable(simIn, this.RoadTimeVar, this.RoadTime);
                        simIn = setVariable(simIn, this.RoadHeightVar, this.RoadHeight);
                        simIn = setVariable(simIn, 'msm_sms', this.msm_sms);
                        simIn = setVariable(simIn, 'mum_umu', this.mum_umu);
                        simIn = setVariable(simIn, 'kt', this.kt);
                        simIn = setVariable(simIn, 'ct', this.ct);

                        simOut = sim(simIn);
                        this.SimTime = stopT;
                        [zs, zus, vs, vus, Ft_val, as_val] = this.readSimSignals(simOut);

                        % ensure scalars
                        zs = double(scalarize(zs));
                        zus = double(scalarize(zus));
                        vs = double(scalarize(vs));
                        vus = double(scalarize(vus));
                        Ft_val = double(scalarize(Ft_val));
                        as_val = double(scalarize(as_val));
                    catch ME
                        % fallback to zeros
                        zs = 0; zus = 0; vs = 0; vus = 0; Ft_val = 0; as_val = 0;
                    end
                else
                    % No model available -> zeros
                    zs = 0; zus = 0; vs = 0; vus = 0; Ft_val = 0; as_val = 0;
                end
            end
        
            % Derived quantities
            deflection = zs - zus;
            defl_vel = vs - vus;
            if idx > 1
                as_prev = this.State(end) * this.Scale_a;
                jerk = (as_val - as_prev) / this.Ts;
            else
                jerk = 0;
            end
        
            delta_k = action(1) - this.PreviousAction(1);
            delta_c = action(2) - this.PreviousAction(2);
        
            na = as_val / this.Scale_a;
            nj = jerk / this.Scale_j;
            nF = Ft_val / this.Scale_F;
            ndx = deflection / this.Scale_dx;
            ndv = defl_vel / this.Scale_dv;
            ndk = delta_k / this.Scale_dk;
            ndc = delta_c / this.Scale_dc;
            sat = 2.0;
            sat_pen = max(0, abs(ndx)-1).^2;
        
            % Compute reward
            w = this.RewardWeights;
            
            cost = w(1)*na^2 + w(2)*nj^2 + w(3)*nF^2 + w(4)*ndv^2 + w(5)*ndx^2 + sat*sat_pen + w(6)*ndk^2 + w(7)*ndc^2;
            localScale = max(eps, this.scaleR);
            reward = 1 - tanh(cost/localScale);
            if abs(ndx)<=1 && abs(nF)<=1 && abs(na)<=1
                reward = reward + 0.02;
            end
            reward = max(min(reward,1),-1);
        
            this.CurrentEpisodeReward = this.CurrentEpisodeReward + reward;
        
            % Update state
            zr = this.safeIndex(this.RadarPreview, idx);
            this.State = [zr; zs/this.Scale_dx; zus/this.Scale_dx; vs/this.Scale_dv; vus/this.Scale_dv; Ft_val/this.Scale_F; as_val/this.Scale_a];
            this.PreviousAction = action(:);
        
            % Logged signals
            loggedSignals = struct('State', this.State, 'Action', action, 'Cost', cost, 'Reward', reward);
        
            % Update episode log
            logIndex = max(1, min((idx - this.StartIdx + 1), this.MaxSteps));
            this.EpisodeLog(logIndex) = struct('State', this.State, 'Action', action, 'Reward', reward);
        
            % Increment step and determine isDone
            this.CurrentStep = this.CurrentStep + 1;
            isDone = ((this.CurrentStep - this.StartIdx) > this.MaxSteps) || (this.CurrentStep > length(this.RadarPreview)) || ((this.CurrentStep*this.Ts) >= this.maxTime);

            % Append episode reward at termination
            if isDone
                this.EpisodeRewardHistory(end+1) = this.CurrentEpisodeReward;
            end
        
            obs = this.State;
        end



        %% readSimSignals: extract last sample values from simOut.logsout
        function [zs, zus, vs, vus, Ft_val, as_val] = readSimSignals(this, simOut)
        
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
                for i = 1:numel(this.LoggedSignalNames)
                    name = this.LoggedSignalNames{i};
        
                    try
                        % Extract signal by name
                        data = logs.get(name);
        
                        % Remove the last entry if more than one sample exists
                        if numel(data) > length(this.RadarPreview)
                            data = data(1:end-1);
                        end
        
                        % Pick the last valid value (after removing last entry)
                        lastVal = data;
        
                    catch
                        % If signal not found or invalid, default to 0
                        lastVal = 0;
                    end
        
                    % Assign to corresponding output variable
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

        %% Simple helpers
        function [kB, cB] = getActionBounds(this)
            kB = [this.ActionInfo.LowerLimit(1), this.ActionInfo.UpperLimit(1)];
            cB = [this.ActionInfo.LowerLimit(2), this.ActionInfo.UpperLimit(2)];
        end

        function s = getState(this), s = this.State; end

        function val = safeIndex(~, v, idx)
            if isempty(v), val = 0; return; end
            if idx <= numel(v), val = v(idx); else val = v(end); end
        end

        function validLog = getValidEpisodeLog(this)
            filled = arrayfun(@(x) ~isempty(x.State), this.EpisodeLog);
            idx = find(filled, 1, 'last');
            if isempty(idx), validLog = []; else validLog = this.EpisodeLog(1:idx); end
        end

        function plotEpisode(this)
            log = this.getValidEpisodeLog();
            if isempty(log), warning('No episode log'); return; end
            zs_vals = arrayfun(@(x)x.State(2), log);
            vs_vals = arrayfun(@(x)x.State(4), log);
            figure; plot(zs_vals); hold on; plot(vs_vals); legend('zs','vs'); title('Episode'); xlabel('Time Step');
        end

        function plotRewardCurve(this, window)
            if nargin < 2, window = 5; end
            rewards = this.EpisodeRewardHistory;
            if isempty(rewards), disp('No reward history'); return; end
            movingAvg = (length(rewards) >= window) * movmean(rewards, window) + (length(rewards) < window) * rewards;
            figure; plot(rewards,'b-'); hold on; plot(movingAvg,'r-','LineWidth',2); xlabel('Episode'); ylabel('Total Reward'); title('Reward Curve');
        end

        function validateLoggedSignals(this)
            % Robust validation of expected logged signal names
            if ~this.SimulinkAvailable
                warning('validateLoggedSignals: SimulinkAvailable is false; skipping validation.');
                return;
            end
        
            % Ensure RoadTime and RoadHeight exist and are non-empty
            if isempty(this.RoadTime) || isempty(this.RoadHeight)
                error('validateLoggedSignals: RoadTime or RoadHeight is empty. Did you call reset() before validation?');
            end
        
            try
                simIn = Simulink.SimulationInput(this.ModelName);
            catch ME
                error('validateLoggedSignals: couldn''t create SimulationInput for model "%s": %s', ...
                    this.ModelName, ME.message);
            end
        
            try
                simIn = setModelParameter(simIn, 'StopTime', num2str(this.Ts));
            catch ME
                error('validateLoggedSignals: failed to set StopTime: %s', ME.message);
            end
        
            try
                simIn = setVariable(simIn, this.RoadTimeVar, this.RoadTime);
                simIn = setVariable(simIn, this.RoadHeightVar, this.RoadHeight);
            catch ME
                error('validateLoggedSignals: failed to set variables RoadTimeVar/RoadHeightVar: %s', ME.message);
            end
        
            try
                simOut = sim(simIn);
            catch ME
                error('validateLoggedSignals: simulation failed: %s', ME.message);
            end
        
            if ~isprop(simOut, 'logsout')
                error('validateLoggedSignals: simOut has no logsout property.');
            end
        
            logs = simOut.logsout;
            missing = {};
            for i = 1:numel(this.LoggedSignalNames)
                sigName = this.LoggedSignalNames{i};
                try
                    logs.get(sigName);
                catch
                    missing{end+1} = sigName; %#ok<AGROW>
                end
            end
            if ~isempty(missing)
                warning('validateLoggedSignals: these LoggedSignalNames are missing from logsout: %s', strjoin(missing, ', '));
            end
        end


    end
end
