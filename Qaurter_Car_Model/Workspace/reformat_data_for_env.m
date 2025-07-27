% radar_data.Properties
% radar = table2array(radar_data);
% radar = radar(:,2);
% save("D:\kedar project\PRISM-RL\Qaurter_Car_Model\samples_test\road_profile_ISO_8608_ClassA\Prediction_data\radarData.mat", "radar")

%zs = out.y_s;
%zus = out.y_u;
%vs = out.y_sdot;
%vus = out.y_udot;
%Ft = out.Ft;
%as = out.y_sddot;
%suspension = [zs, zus, vs, vus, Ft, as];
%save("D:\kedar project\PRISM-RL\Qaurter_Car_Model\samples_test\road_profile_ISO_8608_ClassA\Prediction_data\SuspensionData.mat", "suspension");
radar = radar(1:1000);
