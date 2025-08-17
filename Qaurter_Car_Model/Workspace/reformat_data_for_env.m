radar = radar_data.RadarRoadElevation_m_;
Suspension_curr = [out.y_s out.y_u out.y_sdot out.y_udot out.Ft out.y_sddot];
suspension = Suspension_curr(1:end-1, :);
save('D:\kedar project\PRISM-RL\Qaurter_Car_Model\samples_test\road_profile_ISO_8608_ClassA\Prediction_data\radarData.mat', 'radar')
save('D:\kedar project\PRISM-RL\Qaurter_Car_Model\samples_test\road_profile_ISO_8608_ClassA\Prediction_data\SuspensionData.mat', 'suspension')