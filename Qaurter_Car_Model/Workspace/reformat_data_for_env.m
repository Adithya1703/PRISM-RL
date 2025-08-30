% radar = [radar1;radar2]';
% suspension = [suspension1; suspension2];
% save('C:\Adithya\PRISM-RL\Qaurter_Car_Model\samples_test\main_data\radarData.mat', 'radar')
% save('C:\Adithya\PRISM-RL\Qaurter_Car_Model\samples_test\main_data\SuspensionData.mat', 'suspension')

mdl = 'new_model_Qaurter_Car'; % change if different
toBlocks = find_system(mdl, 'BlockType', 'ToWorkspace');
fprintf('To Workspace blocks:\n');
for i=1:numel(toBlocks)
    varName = get_param(toBlocks{i}, 'VariableName');
    saveFmt = get_param(toBlocks{i}, 'SaveFormat');
    fprintf('  %s  -> VariableName="%s"  SaveFormat="%s"\n', toBlocks{i}, varName, saveFmt);
end