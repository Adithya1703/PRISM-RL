function env = createEnvironment()
    % load you radar and suspension data
    load("D:\kedar project\PRISM-RL\Qaurter_Car_Model\samples_test\road_profile_ISO_8608_ClassA\Prediction_data\radarData.mat","radar"); % should contain radar variable
    load("D:\kedar project\PRISM-RL\Qaurter_Car_Model\samples_test\road_profile_ISO_8608_ClassA\Prediction_data\SuspensionData.mat", "suspension"); % should contain suspension variable

    % define reward weights
    rewardWeights = [1.0, 0.8, 0.3, 0.1, 0.05, 0.01, 0.01];
    envObj = PredictiveSuspensionEnv(radar, suspension, rewardWeights);

    % Wrap it using rlFUnctionEnv
    ObservationEnv = envObj.ObservationInfo;
    ActionInfo = envObj.ActionInfo;

    StepHandle = @(action) step(envObj, action);
    ResetHandle = @() resetAndReturnInitialObs(envObj);

    env = rlFunctionEnv(ObservationEnv, ActionInfo, StepHandle, ResetHandle);
end

function obs = resetAndReturnInitialObs(envObj)
    reset(envObj);
    obs = envObj.State;
end