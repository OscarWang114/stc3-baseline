### All 
baseline_nugget_chinese_HelpDeskType.ALL_Nov-11_07-38-34-910121: ALL
baseline_nugget_chinese_ALL_Nov-12_02-34-52-560410: ALL-40ep

old/nugget_chinese_ALL_test_submission.origin.json
'nugget': {'jsd': 0.022733284284312004, 'rnss': 0.089064352627035173}

### Pre train then Fine tune

baseline_nugget_chinese_LT_Nov-12_02-26-01-271739: ALL-LT-bs64-lr1e-4-10ep
baseline_nugget_chinese_CZ_Nov-12_05-42-34-643161: ALL-CZ-lr1e-4-10ep
baseline_nugget_chinese_DX_Nov-12_05-09-37-867082: ALL-DX-lr1e-4-10ep

baseline_nugget_chinese_CZ_Nov-12_14-38-03-637632: ALL-CZ-lr1e-4-30ep
baseline_nugget_chinese_CZ_Nov-12_14-50-40-401466: ALL-CZ-lr1e-4-10ep

Starting over-fitting from 10-13ep
baseline_nugget_chinese_CZ_Nov-12_14-57-03-696905: ALL-CZ-bs256-lr1e-4-20ep
baseline_nugget_chinese_LT_Nov-12_15-01-09-805619: ALL-LT-bs256-lr1e-4-20ep
baseline_nugget_chinese_DX_Nov-12_15-05-19-846363: ALL-DX-bs256-lr1e-4-20ep

bs256-lr1e-4-ep10 looks good
baseline_nugget_chinese_DX_Nov-12_15-10-34-999909: ALL-DX-bs256-lr1e-4-10ep
baseline_nugget_chinese_CZ_Nov-12_15-12-53-935816: ALL-CZ-bs256-lr1e-4-10ep
baseline_nugget_chinese_LT_Nov-12_15-14-54-059207: ALL-LT-bs256-lr1e-4-10ep


### Pre train then Fine tune - GRU
nugget_chinese_ALL_test_submission.GRU.json
'nugget': {'jsd': 0.02436372681256792, 'rnss': 0.09246081547976193}

CZ
nugget': {'jsd': 0.03489445509547635, 'rnss': 0.10915026758275478}
LT
'nugget': {'jsd': 0.02251959178501944, 'rnss': 0.09157513405699717}
DX
'nugget': {'jsd': 0.02080846373483575, 'rnss': 0.08668402648735195}




### Train from Scratch

baseline_nugget_chinese_LT_Nov-12_01-46-58-066226: LT-30ep-hidden100-layers2
baseline_nugget_chinese_LT_Nov-12_01-53-40-181281: LT-bs32-lr1e-4-30ep-hidden100-layers2
baseline_nugget_chinese_LT_Nov-12_01-58-57-090189: LT-bs32-50ep-hidden100-layers2
baseline_nugget_chinese_LT_Nov-12_02-08-38-770913: LT-50ep