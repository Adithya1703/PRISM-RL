clc;
clear;

diary('logfile.txt');   % Record logs
diary on;

%% 1. Create/load environment
env = createEnvironment();

%% 2. Extract specs
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

obsDim = prod(obsInfo.Dimension); 
actDim = prod(actInfo.Dimension); 

actLB = actInfo.LowerLimit(:);
actUB = actInfo.UpperLimit(:);
Ts = 0.01; 

%% 3. Critic network
obsPath = [
    featureInputLayer(obsDim,"Normalization","none","Name","obs")
    fullyConnectedLayer(256,"Name","c_fc1")
    reluLayer("Name","c_relu1")
    fullyConnectedLayer(256,"Name","c_fc2")
];
actPath = [
    featureInputLayer(actDim,"Normalization","none","Name","act")
    fullyConnectedLayer(256,"Name","a_fc1")
];
common = [
    additionLayer(2,"Name","add")
    reluLayer("Name","c_relu2")
    fullyConnectedLayer(256,"Name","c_fc3")
    reluLayer("Name","c_relu3")
    fullyConnectedLayer(1,"Name","QValue")
];
critLG = layerGraph(obsPath);
critLG = addLayers(critLG, actPath);
critLG = addLayers(critLG, common);
critLG = connectLayers(critLG,"c_fc2","add/in1");
critLG = connectLayers(critLG,"a_fc1","add/in2");

criticNet = dlnetwork(critLG);
critic = rlQValueFunction(criticNet, obsInfo, actInfo, ...
    "ObservationInputNames","obs", ...
    "ActionInputNames","act");

%% 4. Actor network
actorLG = layerGraph([
    featureInputLayer(obsDim,"Normalization","none","Name","obs")
    fullyConnectedLayer(256,"Name","a_fc1")
    reluLayer("Name","a_relu1")
    fullyConnectedLayer(256,"Name","a_fc2")
    reluLayer("Name","a_relu2")
    fullyConnectedLayer(actDim,"Name","a_fcOut")
    tanhLayer("Name","tanh")
]);
scaleFcn = @(x) ((actUB + actLB)/2) + ((actUB - actLB)/2) .* x;
scaleLayer = functionLayer(scaleFcn,"Name","scale","Formattable",true);

actorLG = addLayers(actorLG, scaleLayer);
actorLG = connectLayers(actorLG,"tanh","scale/in");

actorNet = dlnetwork(actorLG);
actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo, ...
    "ObservationInputNames","obs");

%% 5. Agent options (DDPG)
criticOpts = rlOptimizerOptions('LearnRate',1e-4,'GradientThreshold',1);
actorOpts  = rlOptimizerOptions('LearnRate',1e-4,'GradientThreshold',1);

agentOpts = rlDDPGAgentOptions( ...
    "SampleTime", Ts, ...
    "TargetSmoothFactor", 1e-3, ...
    "TargetUpdateFrequency", 1, ...
    "ExperienceBufferLength", 1e6, ...
    "MiniBatchSize", 256, ...
    "DiscountFactor", 0.995, ...
    "ActorOptimizerOptions", actorOpts, ...
    "CriticOptimizerOptions", criticOpts);

range = actUB - actLB;
agentOpts.NoiseOptions.StandardDeviation = 0.10 * range; 
agentOpts.NoiseOptions.StandardDeviationDecayRate = 1e-5;
agentOpts.NoiseOptions.MeanAttractionConstant = 0.15;
agentOpts.NoiseOptions.Mean = zeros(actDim,1); 
agentOpts.NoiseOptions.VarianceMin = (0.02 * range).^2; 

%% 6. Normalizers (IMPORTANT: before agent creation)
obsNrz = rlNormalizer(obsInfo.Dimension,'Normalization','zscore');
actNrz = rlNormalizer(actInfo.Dimension,'Normalization','rescale-symmetric', ...
                      'Min',actLB,'Max',actUB);

actor = setNormalizer(actor, obsNrz);
critic = setNormalizer(critic, [obsNrz, actNrz]);

agent = rlDDPGAgent(actor, critic, agentOpts);

%% 7. Training options
maxSteps = 1000;
trainOpts = rlTrainingOptions( ...
    "MaxEpisodes", 1600, ...
    "MaxStepsPerEpisode", maxSteps, ...
    "ScoreAveragingWindowLength",20, ...
    "StopTrainingCriteria","AverageReward", ...
    "StopTrainingValue",5000, ...   % safer than 1e6
    "SaveAgentCriteria","EpisodeFrequency", ...
    "SaveAgentValue",200, ...       % save every 200 episodes
    "SaveAgentDirectory","agents", ...
    "Verbose",true, ...
    "Plots","training-progress");

%% 8. Train agent
if ~exist('agents','dir'), mkdir('agents'); end
trainingStats = train(agent, env, trainOpts);
save("Agent.mat", "agent")
diary off;
