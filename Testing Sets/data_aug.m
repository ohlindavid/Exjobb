folder = "C:\Users\Oskar\Documents\GitHub\Exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_6";
aim_folder = folder + "\augmented_data\";
direc = dir(folder);
direc = direc(3:end);
mkdir(aim_folder);
for file = {direc.name}
    data = load(folder+"\"+file{1});
    nchan = size(data,2);
    length = size(data,1);
    
    % Subtract mean.
    the_mean = mean(data,1);
    data = data - the_mean;
 
    % Normalize/standardize signals (maybe mean of sum of abs?) 
    mean_of_sum_of_abs = mean(mean(abs(data),1));
    data = data/mean_of_sum_of_abs; 
    
    % Save to folder.
    csvwrite(aim_folder+file,data); 
end

folder = aim_folder;
direc = dir(folder);
direc = direc(3:end);

for file = {direc.name}
    % Invert Signal.
    data = load(folder+"\"+file{1});
    csvwrite(aim_folder+file+"_neg",-data);
end

direc = dir(folder);
direc = direc(3:end);

for file = {direc.name}
    % Noise.
    data = load(folder+"\"+file{1});
    noise = 0.001*randn(length,nchan);
    data_noise = data+noise;
    csvwrite(aim_folder+file+"_noise1",data_noise); 
end

direc = dir(folder);
direc = direc(3:end);

for file = {direc.name}
    % Noise.
    data = load(folder+"\"+file{1});
    noise = 0.001*randn(length,nchan);
    data_noise = data+noise;
    csvwrite(aim_folder+file+"_noise2",data_noise); 
end

direc = dir(folder);
direc = direc(3:end);

