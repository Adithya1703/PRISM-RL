function env = createEnvironment()
    % Load precomputed radar/suspension if you have them; otherwise use empty
    s1 = load('radarData.mat');   % update path if needed
    s2 = load('SuspensionData.mat');

    if isfield(s1,'radar'), radar = s1.radar; else radar = []; end
    if isfield(s2,'suspension'), suspension = s2.suspension; else suspension = []; end

    % Reward weights (example)
    rewardWeights = [0.40, 0.15, 0.15, 0.08, 0.12, 0.03, 0.02];

    % Create environment object (constructor will enable FastRestart if possible)
    envObj = PredictiveSuspensionEnv(radar, suspension, rewardWeights);

    % If you have static suspension/radar, estimate a robust scaleR; otherwise keep default
    if ~isempty(suspension)
        numSamples = 500;
        N = size(suspension,1);
        costs = zeros(numSamples,1);
        for k = 1:numSamples
            action = [rand*(30000-10000)+10000; rand*(3500-500)+500];
            idx = randi([2, max(2, N)]);
            zs = suspension(idx,1); zus = suspension(idx,2); vs = suspension(idx,3); vus = suspension(idx,4);
            Ft = suspension(idx,5); as = suspension(idx,6);
            deflection = zs - zus;
            defl_vel = vs - vus;
            jerk = (as - suspension(max(1,idx-1),6))/envObj.Ts;
            na = as/envObj.Scale_a; nj = jerk/envObj.Scale_j; nF = Ft/envObj.Scale_F;
            ndx = deflection/envObj.Scale_dx; ndv = defl_vel/envObj.Scale_dv;
            ndk = (action(1)-envObj.k)/envObj.Scale_dk; ndc = (action(2)-envObj.c)/envObj.Scale_dc;
            sat_pen = max(0, abs(ndx)-1).^2; sat = 0.05;
            w = rewardWeights;
            costs(k) = w(1)*(na)^2 + w(2)*(nj)^2 + w(3)*(nF)^2 + w(4)*(ndv)^2 + w(5)*(ndx)^2 + sat*sat_pen + w(6)*(ndk)^2 + w(7)*(ndc)^2;
        end
        posCosts = costs(costs>0);
        if isempty(posCosts)
            envObj.scaleR = max(1e-3, median(abs(costs)) + 1e-3);
        else
            envObj.scaleR = max(1e-3, median(posCosts));
        end
        fprintf('Estimated scaleR: %.6g\n', envObj.scaleR);
    else
        fprintf('No static suspension data loaded; leaving envObj.scaleR = %g\n', envObj.scaleR);
    end

    % Wrap env in rlFunctionEnv
    ObservationEnv = getObservationInfo(envObj);
    ActionInfo = getActionInfo(envObj);

    % ResetHandle must return [obs, loggedSignals]
    ResetHandle = @() resetAndReturnInitialObs(envObj);
    StepHandle = @(action, loggedSignals) step(envObj, action);

    env = rlFunctionEnv(ObservationEnv, ActionInfo, StepHandle, ResetHandle);
end
