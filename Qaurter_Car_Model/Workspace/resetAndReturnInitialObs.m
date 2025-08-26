function [obs, loggedSignals] = resetAndReturnInitialObs(envObj)
    reset(envObj);                % class reset handles episode bookkeeping
    obs = envObj.getState();      % 7x1 vector
    loggedSignals = struct();     % empty struct (placeholder)
end
