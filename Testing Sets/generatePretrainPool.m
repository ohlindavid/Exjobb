% Prepared pretraining for a network eventually for subject 6.
list = who;
do = contains(list,"study");
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
            csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\sets\Albin&Damir\AD_pretraining_pool_crop\A' + string(floor(i/3)+1) + "A" + string(j) ,sig(onset:end,:));
        end
        if (mod(i,3)==2) 
            sig = eval(subj).trial{j}';
            csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\sets\Albin&Damir\AD_pretraining_pool_crop\B' + string(floor(i/3)+1) + "B" + string(j) ,sig(onset:end,:));
        end
        if (mod(i,3)==0) 
            sig = eval(subj).trial{j}';
            csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\sets\Albin&Damir\AD_pretraining_pool_crop\C' + string(floor(i/3)+1) + "C" + string(j) ,sig(onset:end,:));
        end
    end
end