time = 2.2;
bin_size = 4;
number_bins = round(time/bin_size);
v_len = length(Subj00_CleanData_study_img_NegALL0.trial{1});

for i=1:200
    signalA = Subj00_CleanData_study_img_NegALL0.trial{i}';
    signalB = Subj00_CleanData_study_img_NeuALL0.trial{i}';
    signalC = Subj00_CleanData_study_img_PosALL0.trial{i}';
    for j=0:number_bins-1
        low_index = j*floor(v_len/number_bins)+1;
        high_index = (j+1)*floor(v_len/number_bins);
        if (high_index > v_len)
            break;
            
            high_index = v_len;
        end
        sig_2_write_A = signalA(1:1700,:);
        sig_2_write_B = signalB(1:1700,:);
        sig_2_write_C = signalC(1:1700,:);
        csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\SterreTest\A' + string(i),sig_2_write_A);
        csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\SterreTest\B' + string(i),sig_2_write_B);
        csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\SterreTest\C' + string(i),sig_2_write_C);
    end
end
% %%
% v_len = length(Subj20_CleanData_test_FA_lexical.trial{1});
% 
% for i=1:28
%     signalA = Subj20_CleanData_test_FA_lexical.trial{i}';
%     signalB = Subj20_CleanData_test_LM_lexical.trial{i}';
%     signalC = Subj20_CleanData_test_OB_lexical.tria l{i}';
%     for j=0:number_bins-1
%         low_index = j*floor(v_len/number_bins)+1;
%         high_index = (j+1)*floor(v_len/number_bins);
%         if (high_index > v_len)
%             break;
%             high_index = v_len;
%         end
%         sig_2_write_A = signalA(onset:high_index,:);
%         sig_2_write_B = signalB(onset:high_index,:);
%         sig_2_write_C = signalC(onset:high_index,:);
%         csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_20_pred\A' + string(i),sig_2_write_A);
%         csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_20_pred\B' + string(i),sig_2_write_B);
%         csvwrite('C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Albin&Damir\AD_data_set_subject_20_pred\C' + string(i),sig_2_write_C);
% 
%     end
% end