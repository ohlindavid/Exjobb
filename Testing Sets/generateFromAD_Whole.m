time = 4;
bin_size = 4;
number_bins = round(time/bin_size);
v_len = length(Subj06_CleanData_study_FA.trial{1});
onset = floor(1.5/4*v_len);

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
        sig_2_write_A = signalA(onset:high_index,25:30);
        sig_2_write_B = signalB(onset:high_index,25:30);
        sig_2_write_C = signalC(onset:high_index,25:30);
        csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_6\A' + string(i),sig_2_write_A);
        csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_6\B' + string(i),sig_2_write_B);
        csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_6\C' + string(i),sig_2_write_C);
    end
end

v_len = length(Subj06_CleanData_test_FA_lexical.trial{1});

for i=1:29
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
        sig_2_write_A = signalA(onset:high_index,25:30);
        sig_2_write_B = signalB(onset:high_index,25:30);
        sig_2_write_C = signalC(onset:high_index,25:30);
        csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_6_pred\A' + string(i),sig_2_write_A);
        csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_6_pred\B' + string(i),sig_2_write_B);
        csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_6_pred\C' + string(i),sig_2_write_C);

    end
end