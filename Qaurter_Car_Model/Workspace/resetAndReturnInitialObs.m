function [obs, loggedSignals] = resetAndReturnInitialObs(envObj)
    reset(envObj);
    obs = envObj.getState();
    loggedSignals = struct();
end
