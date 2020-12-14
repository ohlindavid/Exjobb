time = 4;
bin_size = 4;
number_bins = round(time/bin_size);
v_len = length(Subj06_CleanData_study_FA.trial{1});

for i=1:60
    signalA = Subj06_CleanData_study_FA.trial{i}';
    signalB = Subj06_CleanData_study_LM.trial{i}';
    signalC = Subj06_CleanData_study_OB.trial{i}';
    for j=0:number_bins-1
        low_index = j*floor(v_len/number_bins)+1;
        high_index = (j+1)*floor(v_len/number_bins);
        if (high_index > v_len)
            break;
            high_index = v_len;
        end
        sig_2_write_A = signalA(low_index:high_index,:);
        sig_2_write_B = signalB(low_index:high_index,:);
        sig_2_write_C = signalC(low_index:high_index,:);
        csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_6\A' + string(i),sig_2_write_A(:,26:28));
        csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_6\B' + string(i),sig_2_write_B(:,26:28));
        csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_6\C' + string(i),sig_2_write_C(:,26:28));
    end
end

v_len = length(Subj06_CleanData_test_FA_lexical.trial{1});

for i=1:32
    signalA = Subj06_CleanData_test_FA_lexical.trial{i}';
    signalB = Subj06_CleanData_test_LM_lexical.trial{i}';
    signalC = Subj06_CleanData_test_OB_lexical.trial{i}';
    for j=0:number_bins-1
        low_index = j*floor(v_len/number_bins)+1;
        high_index = (j+1)*floor(v_len/number_bins);
        if (high_index > v_len)
            break;
            high_index = v_len;
        end
        sig_2_write_A = signalA(low_index:high_index,:);
        sig_2_write_B = signalB(low_index:high_index,:);
        sig_2_write_C = signalC(low_index:high_index,:);
        csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_6_pred\A' + string(i),sig_2_write_A(:,26:28));
        csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_6_pred\B' + string(i),sig_2_write_B(:,26:28));
        csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_6_pred\C' + string(i),sig_2_write_C(:,26:28));

    end
end