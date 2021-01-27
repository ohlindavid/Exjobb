% Prepared pretraining for a network eventually for subject 6.
list = who;
sample_length = 10000;
for i=1:length(list)
    subj = list{i};
    if length(eval(subj).trial{1}) < sample_length
        sample_length = length(eval(subj).trial{1});
    end
end

for i=1:length(list)
    subj = list{i};
    for j = 1:length(eval(subj).trial)
        if (mod(i,3)==1) 
            sig = eval(subj).trial{j}';
            csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_pretraining_pool\A' + string(i) + "A" + string(j) ,sig);
        end
        if (mod(i,3)==2) 
            sig = eval(subj).trial{j}';
            csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_pretraining_pool\B' + string(i) + "B" + string(j) ,sig);
        end
        if (mod(i,3)==0) 
            sig = eval(subj).trial{j}';
            csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_pretraining_pool\C' + string(i) + "C" + string(j) ,sig);
        end
    end
end