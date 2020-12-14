time = 4;
bin_size = 0.2;
number_bins = round(time/bin_size);
v_len = length(Subj06_CleanData_study_FA.trial{1});

for i=1:round(number_bins)
    mkdir ("C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_comp_to_Bramao/bin" + string(i))
end

for i=1:60
    signalA = Subj06_CleanData_study_FA.trial{i}';
    signalB = Subj06_CleanData_study_LM.trial{i}';
    for j=0:number_bins-1
        low_index = j*floor(v_len/number_bins)+1;
        high_index = (j+1)*floor(v_len/number_bins);
        if (high_index > v_len)
            break;
            high_index = v_len;
        end
        sig_2_write_A = signalA(low_index:high_index,:);
        sig_2_write_B = signalB(low_index:high_index,:);
        csvwrite('C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_comp_to_Bramao/bin'+string(j+1)+'/Aalbin_damir_' + string(i),sig_2_write_A);
        csvwrite('C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_comp_to_Bramao/bin'+string(j+1)+'/Balbin_damir_' + string(i),sig_2_write_B);
    end
end

time = 4;
bin_size = 0.2;
number_bins = round(time/bin_size);
v_len = length(Subj06_CleanData_test_FA_lexical.trial{1});

for i=1:round(number_bins)
    mkdir ("C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_comp_to_Bramao_Pred/bin" + string(i))
end

for i=1:32
    signalA = Subj06_CleanData_test_FA_lexical.trial{i}';
    signalB = Subj06_CleanData_test_LM_lexical.trial{i}';
    for j=0:number_bins-1
        low_index = j*floor(v_len/number_bins)+1;
        high_index = (j+1)*floor(v_len/number_bins);
        if (high_index > v_len)
            break;
            high_index = v_len;
        end
        sig_2_write_A = signalA(low_index:high_index,:);
        sig_2_write_B = signalB(low_index:high_index,:);
        csvwrite('C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_comp_to_Bramao_Pred/bin'+string(j+1)+'/Aalbin_damir_' + string(i),sig_2_write_A);
        csvwrite('C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_comp_to_Bramao_Pred/bin'+string(j+1)+'/Balbin_damir_' + string(i),sig_2_write_B);
    end
end