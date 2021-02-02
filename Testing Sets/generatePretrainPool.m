% Prepared pretraining for a network eventually for subject 6.
list = who;
do = contains(list,"test");
sample_length = 100000;
for i=1:length(list)
    subj = list{i};
    if length(eval(subj).trial{1}) < sample_length
        sample_length = length(eval(subj).trial{1});
    end
end
onset =  floor(1.5/4*length(eval(list{1}).trial{1}'));

for i=1:length(list)
    if (do(i)==0)
        continue
    end
    subj = list{i};
    for j = 1:length(eval(subj).trial)
        if (mod(i,3)==1) 
            sig = eval(subj).trial{j}';
<<<<<<< Updated upstream
            csvwrite('C:\Users\david\Documents\GitHub\exjobb\Testing Sets\AD_retrieval_transfer_crop\A' + string(subj) + "A" + string(j) ,sig(onset:end,:));
        end
        if (mod(i,3)==2) 
            sig = eval(subj).trial{j}';
            csvwrite('C:\Users\david\Documents\GitHub\exjobb\Testing Sets\AD_retrieval_transfer_crop\B' + string(subj) + "B" + string(j) ,sig(onset:end,:));
        end
        if (mod(i,3)==0) 
            sig = eval(subj).trial{j}';
            csvwrite('C:\Users\david\Documents\GitHub\exjobb\Testing Sets\AD_retrieval_transfer_crop\C' + string(subj) + "C" + string(j) ,sig(onset:end,:));
=======
            csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\sets\Albin&Damir\AD_pretraining_pool_crop_test\A' + string(subj) + "A" + string(j) ,sig(onset:end,:));
        end
        if (mod(i,3)==2) 
            sig = eval(subj).trial{j}';
            csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\sets\Albin&Damir\AD_pretraining_pool_crop_test\B' + string(subj) + "B" + string(j) ,sig(onset:end,:));
        end
        if (mod(i,3)==0) 
            sig = eval(subj).trial{j}';
            csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\sets\Albin&Damir\AD_pretraining_pool_crop_test\C' + string(subj) + "C" + string(j) ,sig(onset:end,:));
>>>>>>> Stashed changes
        end
    end
end