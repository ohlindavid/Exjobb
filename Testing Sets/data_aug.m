% TRAIN,VALIDATION,TEST SPLIT

folder = "C:\Users\Oskar\Documents\GitHub\Exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_6";
aim_folder_train = folder + "\augmented_data_train\";
aim_folder_val = folder + "\augmented_data_val\";
aim_folder_test = folder + "\data_test\";
direc = dir(folder);
direc = direc(3:end);
mkdir(aim_folder_train);
mkdir(aim_folder_val)
mkdir(aim_folder_test);
direc = vertcat({direc.name});
direc = direc(randperm(length(direc)));
direcOriginal = (direc);
for class = ['A','B','C']
    i = 1;
    direc = {};
    for element=direcOriginal
        if regexp(element{1}, regexptranslate('wildcard',class+"*"))
            direc{i} = element;
            i=i+1;
        end
    end
    
    for file = direc(1:floor((4/5)^2*length(direc)))
        data = load(folder+"\"+file{1});
        nchan = size(data,2);
        len = size(data,1);

        % Subtract mean.
        the_mean = mean(data,1);    
        data = data - the_mean;

        % Normalize/standardize signals (maybe mean of sum of abs?) 
        mean_of_sum_of_abs = mean(sqrt(var(data,1)));
        data = data/mean_of_sum_of_abs; 

        % Save to folder.
        csvwrite(aim_folder_train+file,data); 
    end

    for file = direc(1+floor((4/5)^2*length(direc)):floor(4/5*length(direc)))
        data = load(folder+"\"+file{1});
        nchan = size(data,2);
        len = size(data,1);

        % Subtract mean.
        the_mean = mean(data,1);    
        data = data - the_mean;

        % Normalize/standardize signals (maybe mean of sum of abs?) 
        mean_of_sum_of_abs = mean(sqrt(var(data,1)));
        data = data/mean_of_sum_of_abs; 

        % Save to folder.
        csvwrite(aim_folder_val+file,data); 
    end

    for file = direc(1+floor(4/5*length(direc)):end)
        data = load(folder+"\"+file{1});
        nchan = size(data,2);
        len = size(data,1);

        % Subtract mean.
        the_mean = mean(data,1);    
        data = data - the_mean;

        % Normalize/standardize signals (maybe mean of sum of abs?) 
        mean_of_sum_of_abs = mean(sqrt(var(data,1)));
        data = data/mean_of_sum_of_abs; 

        % Save to folder.
        csvwrite(aim_folder_test+file,data); 
    end
end
%% DATA AUGMENTATION.

for aim_folder = [aim_folder_train]
    
    folder = aim_folder;
    direc = dir(folder);
    direc = direc(3:end);
    direc = vertcat({direc.name});
    direc = direc(randperm(length(direc)));

    for file = direc
        % Invert Signal.
        data = load(folder+"\"+file{1});
        csvwrite(aim_folder+file+"_neg",-data);
    end

    direc = dir(folder);
    direc = direc(3:end);
    direc = vertcat({direc.name});
    direc = direc(randperm(length(direc)));
    
    for file = direc
        % Noise.
        data = load(folder+"\"+file{1});
        noise = 0.01*randn(len,nchan);
        data_noise = data+noise;
        csvwrite(aim_folder+file+"_noise1",data_noise); 
    end
end